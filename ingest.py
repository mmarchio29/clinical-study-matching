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
import time

import chromadb
import requests
from openai import OpenAI
from rank_bm25 import BM25Okapi

# ── Config ────────────────────────────────────────────────────────────────────

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
CT_API_BASE      = "https://clinicaltrials.gov/api/v2/studies"
CONDITION        = "type 2 diabetes"
MAX_TRIALS       = 300
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

client = OpenAI(api_key=OPENAI_API_KEY)


# ── 1. Fetch ──────────────────────────────────────────────────────────────────

def fetch_trials(condition: str = CONDITION, max_trials: int = MAX_TRIALS) -> list[dict]:
    trials, next_token = [], None
    print(f"Fetching up to {max_trials} recruiting trials for: '{condition}'")

    while len(trials) < max_trials:
        params = {
            "query.cond": condition,
            "filter.overallStatus": "RECRUITING",
            "pageSize": min(100, max_trials - len(trials)),
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
            ]),
            "format": "json",
        }
        if next_token:
            params["pageToken"] = next_token

        r = requests.get(CT_API_BASE, params=params, timeout=30)
        r.raise_for_status()
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


# ── 2. Parse ──────────────────────────────────────────────────────────────────

def parse_trial(raw: dict) -> dict | None:
    try:
        p     = raw.get("protocolSection", {})
        ident = p.get("identificationModule", {})
        elig  = p.get("eligibilityModule", {})
        crit  = elig.get("eligibilityCriteria", "").strip()
        if not crit:
            return None
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
        }
    except Exception as e:
        print(f"  Parse warning: {e}")
        return None


# ── 3. Chunk ──────────────────────────────────────────────────────────────────

def split_criteria(text: str) -> tuple[str, str]:
    """Split raw criteria into inclusion and exclusion sections."""
    lower = text.lower()
    inc_idx = exc_idx = None
    for m in ["inclusion criteria:", "inclusion criteria\n"]:
        i = lower.find(m)
        if i != -1:
            inc_idx = i + len(m)
            break
    for m in ["exclusion criteria:", "exclusion criteria\n"]:
        i = lower.find(m)
        if i != -1:
            exc_idx = i + len(m)
            break

    if inc_idx is not None and exc_idx is not None:
        if inc_idx < exc_idx:
            return text[inc_idx:exc_idx - 20].strip(), text[exc_idx:].strip()
        else:
            return text[inc_idx:].strip(), text[exc_idx:inc_idx - 20].strip()
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

    base_meta = {
        "nct_id":  nct,
        "title":   title,
        "phases":  ", ".join(trial["phases"]),
        "sponsor": trial["sponsor"],
        "min_age": trial["min_age"],
        "max_age": trial["max_age"],
        "sex":     trial["sex"],
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
        f"Sponsor: {trial['sponsor']}"
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

def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i: i + EMBED_BATCH_SIZE]
        resp  = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend([r.embedding for r in resp.data])
        print(f"  Embedded {min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)}")
    return embeddings


# ── 5. Store in ChromaDB with explicit HNSW settings ─────────────────────────

def build_chroma(chunks: list[dict]) -> chromadb.Collection:
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

    # Fetch
    raw = fetch_trials()
    with open("data/trials_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    # Parse
    parsed = [parse_trial(t) for t in raw]
    parsed = [t for t in parsed if t]
    print(f"Parsed: {len(parsed)} / {len(raw)} trials")

    # Chunk
    chunks = []
    for t in parsed:
        chunks.extend(chunk_trial(t))
    print(f"Chunks: {len(chunks)} total")

    # Save chunks for eval.py to reference
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f, indent=2)

    # Embed + store
    build_chroma(chunks)

    # BM25
    build_bm25(chunks)

    print("\nIngestion complete.")
    print(f"  Trials:  {len(parsed)}")
    print(f"  Chunks:  {len(chunks)}")
    print(f"  HNSW ef_construction={HNSW_EF_CONSTRUCTION}, M={HNSW_M}")


if __name__ == "__main__":
    main()
