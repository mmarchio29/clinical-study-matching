"""
ingest.py — Fetch trials from ClinicalTrials.gov, chunk them,
embed with OpenAI, and store in ChromaDB with HNSW parameters exposed.

Run once (or re-run to refresh):
    python ingest.py

Also builds a BM25 index saved to data/bm25_index.pkl for retriever.py.
"""

import json
import os
import pickle
import re
import time
from abc import ABC, abstractmethod
import tiktoken
import chromadb
import requests
from openai import OpenAI
from rank_bm25 import BM25Okapi

# ── Config ────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
CT_API_BASE      = "https://clinicaltrials.gov/api/v2/studies"
CONDITION        = os.getenv("TRIAL_CONDITION", "").strip()
MAX_TRIALS       = int(os.getenv("MAX_TRIALS", "3000"))
TARGET_CITY      = os.getenv("TRIAL_CITY", "Boston").strip().lower()
TARGET_STATE     = os.getenv("TRIAL_STATE", "MA").strip().lower()
SOURCE_TYPE      = os.getenv("TRIAL_SOURCE_TYPE", "paid_file").lower()
PAID_SOURCE_PATH = os.getenv("PAID_SOURCE_PATH", "./data/paid_trials_source.json")
EMBED_MODEL      = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100
CHROMA_PATH      = "./data/chroma_db"
COLLECTION_NAME  = "clinical_trials"
BM25_PATH        = "./data/bm25_index.pkl"
CHUNKS_PATH      = "./data/chunks.json"

# HNSW parameters — these control the navigable small world graph structure.
# ef_construction: more = better graph quality, slower build (default 100)
# M: edges per node — more = better recall, more memory (default 16)
# Increasing both improves recall@k at the cost of index build time.
HNSW_EF_CONSTRUCTION = 200
HNSW_M               = 32

if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=OPENAI_API_KEY)


# ── 1. Fetch ──────────────────────────────────────────────────────────────────

class TrialSourceAdapter(ABC):
    @abstractmethod
    def fetch(self, max_trials: int = MAX_TRIALS) -> list[dict]:
        """Return a list of normalized trial records."""


def parse_money_range(text: str) -> tuple[float | None, float | None]:
    if not text:
        return None, None
    nums = [float(n.replace(",", "")) for n in re.findall(r"\$?\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)", text)]
    if not nums:
        return None, None
    return min(nums), max(nums)


class ClinicalTrialsGovAdapter(TrialSourceAdapter):
    def fetch_raw_trials(self, condition: str = CONDITION, max_trials: int = MAX_TRIALS) -> list[dict]:
        trials, next_token = [], None
        scope = f"condition='{condition}'" if condition else "all conditions"
        print(f"Fetching up to {max_trials} recruiting trials for: {scope}")

        while len(trials) < max_trials:
            params = {
                "filter.overallStatus": "RECRUITING",
                "pageSize": min(100, max_trials - len(trials)),
                "query.locn": "Boston, Massachusetts",
                "fields": ",".join([
                    "protocolSection.identificationModule.nctId",
                    "protocolSection.identificationModule.briefTitle",
                    "protocolSection.eligibilityModule.eligibilityCriteria",
                    "protocolSection.eligibilityModule.minimumAge",
                    "protocolSection.eligibilityModule.maximumAge",
                    "protocolSection.eligibilityModule.sex",
                    "protocolSection.eligibilityModule.stdAges",
                    "protocolSection.conditionsModule.conditions",
                    "protocolSection.armsInterventionsModule.interventions",
                    "protocolSection.designModule.phases",
                    "protocolSection.sponsorCollaboratorsModule.leadSponsor",
                    "protocolSection.contactsLocationsModule.locations",
                ]),
                "format": "json",
            }
            if condition:
                params["query.cond"] = condition
            if next_token:
                params["pageToken"] = next_token

            r = requests.get(CT_API_BASE, params=params, timeout=30)
            if not r.ok:
                # Surface API payload to make bad field names or filters easy to debug.
                raise requests.HTTPError(
                    f"{r.status_code} {r.reason} for {r.url}\nResponse body: {r.text}",
                    response=r,
                )
            data = r.json()
            studies = data.get("studies", [])
            if not studies:
                break
            trials.extend(studies)
            print(f"  {len(trials)} fetched...")
            next_token = data.get("nextPageToken")
            if not next_token:
                break
            time.sleep(0.4)

        print(f"Total fetched: {len(trials)}")
        return trials

    def fetch(self, max_trials: int = MAX_TRIALS) -> list[dict]:
        raw = self.fetch_raw_trials(max_trials=max_trials)
        parsed = [t for t in (parse_ctgov_trial(r) for r in raw) if t]
        filtered = [t for t in parsed if is_target_location_trial(t)]
        print(
            f"Location filter: kept {len(filtered)} Boston, MA trials "
            f"(from {len(parsed)} parsed, {len(raw)} fetched)"
        )
        return filtered


class PaidFileAdapter(TrialSourceAdapter):
    """
    Primary paid-trial source adapter.
    Expected file: JSON list of records with paid and location fields.
    """
    def __init__(self, path: str = PAID_SOURCE_PATH):
        self.path = path

    def fetch(self, max_trials: int = MAX_TRIALS) -> list[dict]:
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"Paid source file not found at {self.path}. "
                "Provide PAID_SOURCE_PATH with a JSON array of trial records."
            )
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("Paid trial source must be a JSON array.")

        normalized = []
        for rec in raw[:max_trials]:
            t = parse_paid_trial(rec)
            if t:
                normalized.append(t)
        print(f"Loaded {len(normalized)} paid studies from {self.path}")
        return normalized


def fetch_trials(condition: str = CONDITION, max_trials: int = MAX_TRIALS) -> list[dict]:
    """
    Source-agnostic fetch.
    Set TRIAL_SOURCE_TYPE:
      - paid_file (default)
      - ctgov
    """
    if SOURCE_TYPE == "ctgov":
        return ClinicalTrialsGovAdapter().fetch(max_trials=max_trials)
    return PaidFileAdapter().fetch(max_trials=max_trials)


# ── 2. Parse ──────────────────────────────────────────────────────────────────

def parse_ctgov_trial(raw: dict) -> dict | None:
    try:
        p     = raw.get("protocolSection", {})
        ident = p.get("identificationModule", {})
        elig  = p.get("eligibilityModule", {})
        crit  = elig.get("eligibilityCriteria", "").strip()
        if not crit:
            return None
        locations = p.get("contactsLocationsModule", {}).get("locations", [])
        first_loc = locations[0] if locations else {}
        all_cities = sorted({
            str(loc.get("city", "")).strip()
            for loc in locations if str(loc.get("city", "")).strip()
        })
        all_states = sorted({
            str(loc.get("state", "")).strip()
            for loc in locations if str(loc.get("state", "")).strip()
        })
        all_countries = sorted({
            str(loc.get("country", "")).strip()
            for loc in locations if str(loc.get("country", "")).strip()
        })
        location_lines = []
        for loc in locations:
            city = str(loc.get("city", "")).strip()
            state = str(loc.get("state", "")).strip()
            country = str(loc.get("country", "")).strip()
            parts = [p for p in [city, state, country] if p]
            if parts:
                location_lines.append(", ".join(parts))
        comp_min, comp_max = parse_money_range(crit)

        return {
            "nct_id":        ident.get("nctId", ""),
            "title":         ident.get("briefTitle", ""),
            "conditions":    p.get("conditionsModule", {}).get("conditions", []),
            "interventions": [i.get("name", "") for i in
                              p.get("armsInterventionsModule", {}).get("interventions", [])],
            "phases":        p.get("designModule", {}).get("phases", []),
            "sponsor":       p.get("sponsorCollaboratorsModule", {}).get("leadSponsor", {}).get("name", ""),
            "min_age":       elig.get("minimumAge", "N/A"),
            "max_age":       elig.get("maximumAge", "N/A"),
            "sex":           elig.get("sex", "ALL"),
            "std_ages":      elig.get("stdAges", []),
            "criteria_raw":  crit,
            "amount_min":    comp_min,
            "amount_max":    comp_max,
            "currency":      "USD",
            "payment_type":  "unknown",
            "comp_text":     "",
            "site_city":     first_loc.get("city", ""),
            "site_state":    first_loc.get("state", ""),
            "site_country":  first_loc.get("country", ""),
            "all_site_cities": ", ".join(all_cities),
            "all_site_states": ", ".join(all_states),
            "all_site_countries": ", ".join(all_countries),
            "all_locations_text": " | ".join(location_lines),
            "lat":           None,
            "lon":           None,
            "remote_allowed": False,
            "travel_required": True,
            "status":        "RECRUITING",
            "last_verified": "",
            "contactability": "unknown",
        }
    except Exception as e:
        print(f"  Parse warning: {e}")
        return None


def parse_paid_trial(raw: dict) -> dict | None:
    """
    Normalize paid-source records into common schema.
    """
    nct_id = raw.get("nct_id") or raw.get("study_id") or raw.get("id") or ""
    title = raw.get("title") or raw.get("brief_title") or ""
    criteria_raw = (raw.get("eligibility_criteria") or raw.get("criteria_raw") or "").strip()
    if not nct_id or not title or not criteria_raw:
        return None

    amt_min = raw.get("amount_min")
    amt_max = raw.get("amount_max")
    if amt_min is None or amt_max is None:
        text_min, text_max = parse_money_range(raw.get("comp_text", ""))
        amt_min = amt_min if amt_min is not None else text_min
        amt_max = amt_max if amt_max is not None else text_max

    return {
        "nct_id": nct_id,
        "title": title,
        "conditions": raw.get("conditions", []),
        "interventions": raw.get("interventions", []),
        "phases": raw.get("phases", []),
        "sponsor": raw.get("sponsor", ""),
        "min_age": raw.get("min_age", "N/A"),
        "max_age": raw.get("max_age", "N/A"),
        "sex": raw.get("sex", "ALL"),
        "std_ages": raw.get("std_ages", []),
        "criteria_raw": criteria_raw,
        "amount_min": amt_min,
        "amount_max": amt_max,
        "currency": raw.get("currency", "USD"),
        "payment_type": raw.get("payment_type", "unspecified"),
        "comp_text": raw.get("comp_text", ""),
        "site_city": raw.get("site_city", ""),
        "site_state": raw.get("site_state", ""),
        "site_country": raw.get("site_country", ""),
        "all_site_cities": raw.get("all_site_cities", ""),
        "all_site_states": raw.get("all_site_states", ""),
        "all_site_countries": raw.get("all_site_countries", ""),
        "all_locations_text": raw.get("all_locations_text", ""),
        "lat": raw.get("lat"),
        "lon": raw.get("lon"),
        "remote_allowed": bool(raw.get("remote_allowed", False)),
        "travel_required": bool(raw.get("travel_required", True)),
        "status": raw.get("status", "RECRUITING"),
        "last_verified": raw.get("last_verified", ""),
        "contactability": raw.get("contactability", "unknown"),
    }


def parse_trial(raw: dict) -> dict | None:
    # Backward-compatible entry point
    if "protocolSection" in raw:
        return parse_ctgov_trial(raw)
    return parse_paid_trial(raw)


def is_target_location_trial(trial: dict) -> bool:
    cities   = {c.strip().lower() for c in str(trial.get("all_site_cities", "")).split(",") if c.strip()}
    states   = {s.strip().lower() for s in str(trial.get("all_site_states", "")).split(",") if s.strip()}
    loc_text = str(trial.get("all_locations_text", "")).lower()

    state_aliases = {"ma": {"ma", "massachusetts"}, "massachusetts": {"ma", "massachusetts"}}
    ts = TARGET_STATE.lower().strip()
    target_state_variants = state_aliases.get(ts, {ts})

    city_match  = (not TARGET_CITY)  or (TARGET_CITY in cities)  or (TARGET_CITY in loc_text)
    state_match = (not TARGET_STATE) or bool(states.intersection(target_state_variants)) or any(v in loc_text for v in target_state_variants)
    return city_match and state_match


# ── 3. Chunk ──────────────────────────────────────────────────────────────────

def split_criteria(text: str) -> tuple[str, str]:
    lower = text.lower()
    inc_idx = exc_idx = None
    inc_header_start = exc_header_start = None

    for m in ["inclusion criteria:", "inclusion criteria\n"]:
        i = lower.find(m)
        if i != -1:
            inc_header_start = i
            inc_idx = i + len(m)
            break
    for m in ["exclusion criteria:", "exclusion criteria\n"]: #remove this part 
        i = lower.find(m)
        if i != -1:
            exc_header_start = i
            exc_idx = i + len(m)
            break

    if inc_idx is not None and exc_idx is not None:
        if inc_idx < exc_idx:
            return text[inc_idx:exc_header_start].strip(), text[exc_idx:].strip()
        else:
            return text[inc_idx:].strip(), text[exc_idx:inc_header_start].strip()
    elif inc_idx is not None:
        return text[inc_idx:].strip(), ""
    elif exc_idx is not None:
        return "", text[exc_idx:].strip()
    return text.strip(), ""


def chunk_trial(trial: dict) -> list[dict]:
    """
    Produce 3 chunk types per trial:
      - inclusion: the inclusion criteria text
      - exclusion: the exclusion criteria text
      - metadata:  overview (conditions, interventions, age range, phases)

    Keeping these separate lets HyDE queries target the right chunk type.
    """
    nct   = trial["nct_id"]
    title = trial["title"]
    inc, exc = split_criteria(trial["criteria_raw"])
    if not inc:
        print(f"  Warning: no inclusion criteria parsed for {trial['nct_id']}")

    base_meta = {
        "nct_id":  nct,
        "title":   title,
        "phases":  ", ".join(trial["phases"]),
        "sponsor": trial["sponsor"],
        "min_age": trial["min_age"],
        "max_age": trial["max_age"],
        "sex":     trial["sex"],
        "amount_min": trial.get("amount_min"),
        "amount_max": trial.get("amount_max"),
        "currency": trial.get("currency", "USD"),
        "payment_type": trial.get("payment_type", ""),
        "site_city": trial.get("site_city", ""),
        "site_state": trial.get("site_state", ""),
        "site_country": trial.get("site_country", ""),
        "all_site_cities": trial.get("all_site_cities", ""),
        "all_site_states": trial.get("all_site_states", ""),
        "all_site_countries": trial.get("all_site_countries", ""),
        "all_locations_text": trial.get("all_locations_text", ""),
        "lat": trial.get("lat"),
        "lon": trial.get("lon"),
        "remote_allowed": bool(trial.get("remote_allowed", False)),
        "travel_required": bool(trial.get("travel_required", True)),
        "status": trial.get("status", "RECRUITING"),
        "last_verified": trial.get("last_verified", ""),
        "contactability": trial.get("contactability", "unknown"),
    }

    chunks = []
    if inc:
        chunks.append({
            "id":   f"{nct}_inclusion",
            "text": f"INCLUSION CRITERIA — {title} ({nct}):\n{inc}",
            "metadata": {**base_meta, "type": "inclusion"},
        })
    if exc:
        chunks.append({
            "id":   f"{nct}_exclusion",
            "text": f"EXCLUSION CRITERIA — {title} ({nct}):\n{exc}",
            "metadata": {**base_meta, "type": "exclusion"},
        })

    meta_text = (
        f"TRIAL OVERVIEW — {title} ({nct})\n"
        f"Conditions: {', '.join(trial['conditions'])}\n"
        f"Interventions: {', '.join(trial['interventions'])}\n"
        f"Phases: {', '.join(trial['phases'])}\n"
        f"Age: {trial['min_age']} – {trial['max_age']}  |  Sex: {trial['sex']}\n"
        f"Sponsor: {trial['sponsor']}\n"
        f"Compensation: {trial.get('amount_min')} - {trial.get('amount_max')} {trial.get('currency', 'USD')} ({trial.get('payment_type', '')})\n"
        f"Location: {trial.get('site_city', '')}, {trial.get('site_state', '')}, {trial.get('site_country', '')}\n"
        f"All sites: {trial.get('all_locations_text', '')}\n"
        f"Remote allowed: {trial.get('remote_allowed', False)} | Travel required: {trial.get('travel_required', True)}\n"
        f"Status: {trial.get('status', 'RECRUITING')} | Contactability: {trial.get('contactability', 'unknown')}"
    )
    chunks.append({
        "id":   f"{nct}_metadata",
        "text": meta_text,
        "metadata": {
            **base_meta,
            "type":         "metadata",
            "conditions":   ", ".join(trial["conditions"]),
            "interventions": ", ".join(trial["interventions"]),
        },
    })
    return chunks


# ── 4. Embed ──────────────────────────────────────────────────────────────────

EMBED_MAX_TOKENS = 8192
_tokenizer = tiktoken.get_encoding("cl100k_base")

def truncate_to_token_limit(text: str, max_tokens: int = EMBED_MAX_TOKENS) -> str:
    tokens = _tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    print(f"  Truncating chunk from {len(tokens)} → {max_tokens} tokens")
    return _tokenizer.decode(tokens[:max_tokens])

def embed_texts(texts: list[str]) -> list[list[float]]:  # replace existing function
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i: i + EMBED_BATCH_SIZE]
        batch = [truncate_to_token_limit(t) for t in batch]
        resp  = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([r.embedding for r in resp.data])
        print(f"  Embedded {min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)}")
    return embeddings

# ── 5. Store in ChromaDB with explicit HNSW settings ─────────────────────────

def build_chroma(chunks: list[dict]) -> chromadb.Collection:
    if not chunks:
        raise ValueError(
            "No chunks to index. Your current filters returned zero studies. "
            "Try increasing MAX_TRIALS or adjusting TRIAL_CITY/TRIAL_STATE."
        )
    os.makedirs(CHROMA_PATH, exist_ok=True)
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    # hnsw:ef_construction and hnsw:M control the navigable small world graph.
    # Higher values = better recall at query time, slower index build.
    collection = chroma.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space":           "cosine",
            "hnsw:construction_ef": HNSW_EF_CONSTRUCTION,
            "hnsw:M":               HNSW_M,
        },
    )

    texts      = [c["text"]     for c in chunks]
    ids        = [c["id"]       for c in chunks]
    metadatas  = [c["metadata"] for c in chunks]

    print(f"\nEmbedding {len(chunks)} chunks...")
    embeddings = embed_texts(texts)

    for i in range(0, len(chunks), 500):
        collection.upsert(
            ids        = ids[i: i + 500],
            embeddings = embeddings[i: i + 500],
            documents  = texts[i: i + 500],
            metadatas  = metadatas[i: i + 500],
        )
    print(f"ChromaDB: {len(chunks)} chunks stored (HNSW ef={HNSW_EF_CONSTRUCTION}, M={HNSW_M})")
    return collection


# ── 6. Build BM25 index ───────────────────────────────────────────────────────

def build_bm25(chunks: list[dict]) -> None:
    """
    Tokenise every chunk and build a BM25Okapi index.
    Saved alongside a chunk-ID list so retriever.py can map
    BM25 result positions back to chunk IDs.
    """
    if not chunks:
        raise ValueError(
            "No chunks available for BM25 index. "
            "Check your location filter or ingestion scope."
        )
    tokenised = [c["text"].lower().split() for c in chunks]
    bm25      = BM25Okapi(tokenised)
    chunk_ids = [c["id"] for c in chunks]

    os.makedirs("data", exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": chunk_ids}, f)
    print(f"BM25 index: {len(chunks)} docs saved to {BM25_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    # Fetch (already returns normalized, parsed records)
    trials = fetch_trials()
    with open("data/trials_raw.json", "w") as f:
        json.dump(trials, f, indent=2)

    print(f"Fetched and parsed: {len(trials)} trials")

    # Chunk
    chunks = []
    for t in trials:
        chunks.extend(chunk_trial(t))
    print(f"Chunks: {len(chunks)} total")
    if not chunks:
        raise RuntimeError(
            "No chunks produced from filtered trials. "
            "Try: TRIAL_CITY=Boston, TRIAL_STATE=Massachusetts, MAX_TRIALS=5000."
        )

    # Save chunks for eval.py to reference
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    # Embed + store
    build_chroma(chunks)

    # BM25
    build_bm25(chunks)

    print("\nIngestion complete.")
    print(f"  Trials:  {len(trials)}")
    print(f"  Chunks:  {len(chunks)}")
    print(f"  HNSW ef_construction={HNSW_EF_CONSTRUCTION}, M={HNSW_M}")

if __name__ == "__main__":
    main()
