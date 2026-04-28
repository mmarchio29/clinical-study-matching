"""
retriever.py — The full RAG pipeline. This is where the retrieval
techniques live. Demonstrates:

  1. HyDE  — Hypothetical Document Embeddings
             Instead of embedding the patient profile directly, ask GPT-4o
             to generate what the *ideal matching trial's criteria* would
             look like. Embed that hypothetical doc. This aligns the query
             vector with the document space rather than the patient space.

  2. ANN   — Approximate Nearest Neighbor via ChromaDB's HNSW index.
             The ef_search parameter controls the recall/speed tradeoff at
             query time (higher ef = more nodes explored = better recall).

  3. BM25  — Keyword search on the same corpus. Runs in parallel with ANN.
             Acts as a hard-term-matching complement to semantic search.

  4. RRF   — Reciprocal Rank Fusion. Merges ANN + BM25 ranked lists without
             needing to calibrate scores across different spaces.
             score(d) = Σ 1 / (k + rank(d))  for each retriever.

  5. Agent loop — GPT-4o checks eligibility criterion-by-criterion, with
             re-retrieval when confidence is below threshold.

Usage:
    python retriever.py                   # runs demo patient
    from retriever import retrieve, run_agent
"""

import json
import os
import pickle
import re
from collections import defaultdict

import chromadb
from openai import OpenAI
import chromadb

# ── Config ────────────────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
CHROMA_PATH     = "./data/chroma_db"
COLLECTION_NAME = "clinical_trials"
BM25_PATH       = "./data/bm25_index.pkl"
EMBED_MODEL     = "text-embedding-3-small"
AGENT_MODEL     = "gpt-4o"

TOP_K             = 40   # candidates per retriever before fusion
FINAL_K           = 10    # top results after RRF fusion to pass to agent
RRF_K             = 60   # RRF constant (standard value)
EF_SEARCH         = 100  # HNSW ef at query time — sweep this in eval.py
CONFIDENCE_THRESH = 0.75 # below this → re-retrieve (max 2 retries)
MAX_RETRIES       = 2

client = OpenAI(api_key=OPENAI_API_KEY)

# ── Shared state (loaded once) ────────────────────────────────────────────────

_collection = None
_bm25_data  = None

def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        chroma      = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = chroma.get_collection(COLLECTION_NAME)
    return _collection

def get_bm25():
    global _bm25_data
    if _bm25_data is None:
        with open(BM25_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data["bm25"], _bm25_data["chunk_ids"]


def _age_to_years(age_str: str | None) -> float | None:
    if not age_str or age_str == "N/A":
        return None
    s = str(age_str).strip().lower()
    m = re.match(r"([0-9]+(?:\.[0-9]+)?)\s*(year|years|month|months|week|weeks|day|days)", s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit.startswith("year"):
        return val
    if unit.startswith("month"):
        return val / 12.0
    if unit.startswith("week"):
        return val / 52.0
    return val / 365.0


def is_active_or_recruiting(metadata: dict) -> bool:
    status = str(metadata.get("status", "RECRUITING")).strip().upper()
    return status.startswith("RECRUITING") or status.startswith("ACTIVE")


# ── Embed helper ──────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    return client.embeddings.create(model=EMBED_MODEL, input=[text]).data[0].embedding


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 1 — HyDE (Hypothetical Document Embeddings)
# ══════════════════════════════════════════════════════════════════════════════

HYDE_SYSTEM = """You are a clinical trial eligibility criteria writer.

Given a patient profile, write what the ideal matching trial's eligibility
criteria would look like. Be specific — include condition names, severity
levels, age ranges, and key exclusion factors.
Write 80-120 words minimum. Use bullet points. No narrative. No explanation.

Do not include location criteria — all trials are in Boston, MA.

Example format:
- Diagnosis of [condition] confirmed by [standard]
- Age [X]-[Y] years
- [Key lab or severity threshold]
- No history of [key exclusion]
- Willing to attend [X] visits"""

DIAGNOSIS_EXPAND_SYSTEM = """You are a clinical terminology expert.
Given a lay description of a health condition, return a JSON list of
formal clinical terms that should be used to search for matching trials.
Return JSON only: {"terms": ["term1", "term2", ...]}
Include: formal diagnosis names, ICD-10 descriptors, related conditions,
and any common trial search terms. Max 6 terms."""

def expand_diagnosis(diagnosis: str) -> str:
    """
    Expand lay terms to clinical terminology before HyDE generation.
    'anxiety' -> 'generalized anxiety disorder, GAD, social anxiety disorder,
                  panic disorder, anxiety NOS, DSM-5 anxiety'
    """
    if not diagnosis or diagnosis == "Healthy volunteer":
        return diagnosis
    resp = client.chat.completions.create(
        model=AGENT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DIAGNOSIS_EXPAND_SYSTEM},
            {"role": "user", "content": diagnosis},
        ],
        temperature=0,
    )
    terms = json.loads(resp.choices[0].message.content).get("terms", [diagnosis])
    expanded = ", ".join(terms)
    return expanded

def generate_hypothetical_doc(patient: dict) -> str:
    """
    HyDE step: ask GPT-4o to hallucinate what a perfectly matching
    trial's eligibility criteria would look like.
    We embed THIS text, not the patient profile.

    Why this works: the patient profile and the trial criteria live in
    different vocabulary spaces. HyDE bridges that gap by generating
    text in the *document* space (criteria language) from the *query*
    space (patient language).
    """
    resp = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": HYDE_SYSTEM},
            {"role": "user",   "content": json.dumps(patient, indent=2)},
        ],
        temperature=0.1,   # slight temperature for natural criteria language
        max_tokens=1000,
    )
    return resp.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 2 — ANN via HNSW (ChromaDB)
# ══════════════════════════════════════════════════════════════════════════════
def build_chroma_where_filter(extracted: dict) -> dict | None:
    filters = []
    sex = extracted.get("sex")
    if sex:
        normalized = sex.upper()
        filters.append({"$or": [
            {"sex": {"$eq": normalized}},
            {"sex": {"$eq": "ALL"}},
        ]})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def age_filter(results: list[dict], age: int | None) -> list[dict]:
    if not age:
        return results
    filtered = []
    for r in results:
        meta = r.get("metadata", {})
        min_age = _age_to_years(meta.get("min_age"))
        max_age = _age_to_years(meta.get("max_age"))
        if min_age is not None and age < min_age:
            continue
        if max_age is not None and age > max_age:
            continue
        filtered.append(r)
    return filtered

def retrieve_ann(
    query_embedding: list[float],
    top_k: int = TOP_K,
    chunk_type: str | None = None,
    where_filter: dict | None = None,
) -> list[dict]:
    """
    Approximate Nearest Neighbor search using ChromaDB's HNSW index.

    ef_search controls how many nodes the HNSW graph traversal explores.
    Higher ef_search → better recall, slower query.
    This is the 'ef' parameter in the navigable small worlds algorithm:
    the search starts at entry nodes and greedily navigates the graph,
    keeping a candidate list of size ef_search at each step.

    We expose ef_search so eval.py can sweep it and plot recall vs latency.
    """
    collection = get_collection()

    if chunk_type and where_filter:
        where = {"$and": [{"type": {"$eq": chunk_type}}, where_filter]}
    elif chunk_type:
        where = {"type": chunk_type}
    else:
        where = where_filter

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results        = top_k,
        where            = where,
        include          = ["documents", "metadatas", "distances"]
    )

    return [
        {
            "id":       results["metadatas"][0][i].get("nct_id", "") + "_" +
                        results["metadatas"][0][i].get("type", ""),
            "nct_id":   results["metadatas"][0][i].get("nct_id", ""),
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score":    1 - results["distances"][0][i],   # cosine sim
            "source":   "ann",
        }
        for i in range(len(results["documents"][0]))
    ]


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 3 — BM25 keyword search
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_bm25(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    BM25 (Best Match 25) keyword search over the same chunk corpus.

    BM25 scores documents by term frequency (TF) weighted by inverse
    document frequency (IDF), with length normalisation.
    Unlike semantic search, it requires exact or near-exact term overlap.

    This is our ablation baseline: if BM25 recall@k ≈ ANN recall@k,
    semantic search isn't adding much. If ANN >> BM25, it proves the
    embedding space captures meaning beyond keyword matching.
    """
    bm25, chunk_ids = get_bm25()
    tokens  = query.lower().split()
    scores  = bm25.get_scores(tokens)

    # Get top_k indices sorted by score descending
    ranked  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    # Load full chunk texts to return (we need the collection for text)
    collection = get_collection()
    results = []
    for rank_pos, idx in enumerate(ranked):
        cid = chunk_ids[idx]
        #if scores[idx] == 0:
            #break   # no term overlap at all — stop early

        # Fetch the chunk from ChromaDB by ID
        fetched = collection.get(ids=[cid], include=["documents", "metadatas"])
        if not fetched["documents"]:
            continue
        meta = fetched["metadatas"][0]
        results.append({
            "id":       cid,
            "nct_id":   meta.get("nct_id", ""),
            "text":     fetched["documents"][0],
            "metadata": meta,
            "score":    float(scores[idx]),
            "source":   "bm25",
        })
    return results


def filter_active_results(results: list[dict]) -> list[dict]:
    return [r for r in results if is_active_or_recruiting(r.get("metadata", {}))]


# ══════════════════════════════════════════════════════════════════════════════
# TECHNIQUE 4 — Reciprocal Rank Fusion
# ══════════════════════════════════════════════════════════════════════════════

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = RRF_K,
) -> list[dict]:
    """
    RRF merges multiple ranked lists without needing to normalise scores
    across different spaces (cosine similarity vs BM25 score have no
    common scale).

    For each document d and each ranked list r:
        RRF_score(d) = Σ  1 / (k + rank_r(d))

    k=60 is the standard default (from Cormack et al. 2009).
    Documents appearing in both lists get additive boosts.
    Documents only in one list still get a score.

    Returns a merged, re-ranked list of unique chunk results.
    """
    scores: dict[str, float] = defaultdict(float)
    docs:   dict[str, dict]  = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = doc["id"]
            scores[doc_id] += 1.0 / (k + rank)
            if doc_id not in docs:
                docs[doc_id] = doc

    fused = sorted(docs.values(), key=lambda d: scores[d["id"]], reverse=True)
    for doc in fused:
        doc["rrf_score"] = round(scores[doc["id"]], 6)
    return fused

def aggregate_by_trial(fused_chunks: list[dict]) -> list[dict]:
    trial_scores = defaultdict(float)
    trial_docs = {}

    for chunk in fused_chunks:
        nct = chunk["nct_id"]
        trial_scores[nct] += chunk.get("rrf_score", 0)

        if nct not in trial_docs:
            trial_docs[nct] = {
                "nct_id": nct,
                "chunks": [],
                "metadata": chunk["metadata"],
            }
        trial_docs[nct]["chunks"].append(chunk)

    results = []
    for nct, score in trial_scores.items():
        doc = trial_docs[nct]
        doc["score"] = score
        doc["base_score"] = score
        doc["bonus_score"] = 0.0
        results.append(doc)

    return sorted(results, key=lambda x: x["score"], reverse=True)
# ══════════════════════════════════════════════════════════════════════════════
# Main retrieval function (HyDE + ANN + BM25 + RRF)
# ══════════════════════════════════════════════════════════════════════════════

def retrieve(
    patient: dict,
    top_k: int = TOP_K,
    final_k: int = FINAL_K,
    use_hyde: bool = True,
    use_bm25: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Full hybrid retrieval pipeline for one patient.

    Returns:
        {
          "hypothetical_doc": str,      # the HyDE-generated criteria text
          "ann_results":  [chunks],     # raw ANN hits
          "bm25_results": [chunks],     # raw BM25 hits
          "fused_results": [chunks],    # RRF-merged, top final_k
          "query_used": str,            # what was actually embedded
        }
    """
    # ── Step 1: build query ──────────────────────────────────────────────────
    if use_hyde:
        if verbose:
            print("  [HyDE] Generating hypothetical eligibility criteria...")
        hypo_doc = generate_hypothetical_doc(patient)
        query_text = hypo_doc
        if verbose:
            print(f"  [HyDE] Generated ({len(hypo_doc.split())} words)")
    else:
        # Fallback: direct embedding of a plain-text patient summary
        query_text = (
            f"{patient.get('primary_diagnosis', '')} patient. "
            f"Age {patient.get('age', '')}. "
            f"HbA1c {patient.get('lab_values', {}).get('HbA1c', '')}. "
            f"eGFR {patient.get('lab_values', {}).get('eGFR', '')}. "
            f"Medications: {', '.join(patient.get('current_medications', []))}."
        )
        hypo_doc = None

    # ── Step 2: ANN search ───────────────────────────────────────────────────
    query_emb  = embed(query_text)
    ann_hits   = retrieve_ann(query_emb, top_k=top_k)
    ann_hits   = filter_active_results(ann_hits)
    if verbose:
        top_ann = ann_hits[0]["score"] if ann_hits else 0.0
        print(f"  [ANN] Top score: {top_ann:.3f}  ({len(ann_hits)} hits)")

    # ── Step 3: BM25 search (on the hypothetical doc or patient summary) ─────
    if use_bm25:
        bm25_hits = retrieve_bm25(query_text, top_k=top_k)
        bm25_hits = filter_active_results(bm25_hits)
        if verbose:
            top_bm25 = bm25_hits[0]["score"] if bm25_hits else 0.0
            print(f"  [BM25] Top score: {top_bm25:.2f}  ({len(bm25_hits)} hits)")
    else:
        bm25_hits = []

    # ── Step 4: RRF fusion ───────────────────────────────────────────────────
    lists_to_fuse = [ann_hits]
    if bm25_hits:
        lists_to_fuse.append(bm25_hits)
    # fused = reciprocal_rank_fusion(lists_to_fuse)[:final_k]
    fused_chunks = reciprocal_rank_fusion(lists_to_fuse)
    trial_ranked = aggregate_by_trial(fused_chunks)
    top_trials = trial_ranked[:final_k]
    if verbose:
        print("Top trials:", [t["nct_id"] for t in top_trials])
        print("Num fused chunks:", len(fused_chunks))
        print("Num trials after aggregation:", len(trial_ranked))

    return {
        "hypothetical_doc": hypo_doc,
        "query_used":       query_text,
        "ann_results":      ann_hits,
        "bm25_results":     bm25_hits,
        "fused_results": [chunk for trial in top_trials for chunk in trial["chunks"]],
        "top_trials": top_trials,
    }

# ══════════════════════════════════════════════════════════════════════════════
# Agent eligibility loop
# ══════════════════════════════════════════════════════════════════════════════

EVAL_SYSTEM = """You are a clinical trial eligibility specialist.

Given a patient profile and retrieved eligibility criteria chunks for a trial,
evaluate whether the patient meets each criterion.

Each chunk is labeled with a CHUNK_ID. You MUST cite the CHUNK_ID that supports
each criterion check in the cited_chunk_id field. If a criterion is not covered
by any chunk, set cited_chunk_id to null — but this should be rare.

Important: Patient profiles from self-reported intake may lack formal clinical
confirmation. In these cases mark criteria as 'uncertain' rather than
'does_not_meet' — the patient may meet the criterion but hasn't provided
clinical documentation.

Only mark 'not_eligible' when the patient clearly and definitively fails a
criterion (wrong age range, explicitly excluded condition, disqualifying
medication).

All trials are in Boston, MA — do not evaluate location as a criterion.
Do not evaluate willingness to attend or visit frequency.
If exclusion_flags contains explicit true/false values, treat them as
confirmed facts — do not mark these as uncertain.

Return a JSON object with EXACTLY this structure:
{
  "nct_id": "...",
  "trial_title": "...",
  "verdict": "eligible" | "not_eligible" | "uncertain",
  "confidence": 0.0-1.0,
  "criteria_checks": [
    {
      "criterion": "brief criterion description",
      "patient_value": "what the patient has or reported",
      "result": "meets" | "does_not_meet" | "uncertain",
      "reasoning": "1-2 sentences citing the evidence",
      "cited_chunk_id": "the CHUNK_ID this criterion came from, or null"
    }
  ],
  "missing_information": ["list of gaps that prevented a definitive verdict"],
  "summary": "2-3 sentence plain English verdict"
}"""

REWRITE_SYSTEM = """Rewrite this clinical trial search query to be more specific.
Return JSON: {"rewritten_query": "..."}"""

CHAT_RETRIEVAL_SYSTEM = """You rewrite user chat into a clinical trial retrieval query.

Return JSON only:
{
  "search_query": "... concise query for trial retrieval ..."
}

Focus on:
- condition or healthy volunteer intent
- age/sex if present
- location constraints if present
- preferences (paid, remote, travel distance) if present
"""

CHAT_RESPONSE_SYSTEM = """You are a clinical trial matching assistant.

You are given retrieved trial chunks and a user's message.
Rules:
- Use only provided retrieved trial evidence.
- If evidence is insufficient, say so and ask a concise follow-up.
- Prefer plain conversational language.
- Do not claim guaranteed eligibility.
- Mention NCT IDs when recommending studies.
"""


def retrieve_from_query(
    query_text: str,
    top_k: int = TOP_K,
    final_k: int = FINAL_K,
    use_bm25: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Conversational retrieval path (no required patient JSON schema).
    """
    query_emb = embed(query_text)
    ann_hits = filter_active_results(retrieve_ann(query_emb, top_k=top_k))
    if use_bm25:
        bm25_hits = filter_active_results(retrieve_bm25(query_text, top_k=top_k))
    else:
        bm25_hits = []

    lists_to_fuse = [ann_hits]
    if bm25_hits:
        lists_to_fuse.append(bm25_hits)
    fused_chunks = reciprocal_rank_fusion(lists_to_fuse)
    trial_ranked = aggregate_by_trial(fused_chunks)
    top_trials = trial_ranked[:final_k]

    if verbose:
        print("Chat retrieval top trials:", [t["nct_id"] for t in top_trials])

    return {
        "query_used": query_text,
        "ann_results": ann_hits,
        "bm25_results": bm25_hits,
        "fused_results": [chunk for trial in top_trials for chunk in trial["chunks"]],
        "top_trials": top_trials,
    }


def chat_search_and_answer(
    user_message: str,
    conversation_context: str = "",
    top_k: int = TOP_K,
    final_k: int = FINAL_K,
) -> dict:
    """
    End-to-end conversational RAG:
      user chat -> retrieval query rewrite -> hybrid retrieval -> grounded response
    """
    rw = client.chat.completions.create(
        model=AGENT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CHAT_RETRIEVAL_SYSTEM},
            {"role": "user", "content": json.dumps({
                "conversation_context": conversation_context,
                "user_message": user_message,
            }, indent=2)},
        ],
        temperature=0,
    )
    search_query = json.loads(rw.choices[0].message.content).get("search_query", user_message)
    retrieved = retrieve_from_query(search_query, top_k=top_k, final_k=final_k, use_bm25=True, verbose=False)

    evidence_chunks = retrieved.get("fused_results", [])[:20]
    chunk_text = "\n\n---\n\n".join(
        f"[{c.get('id','')}]\n{c.get('text','')}" for c in evidence_chunks
    )

    resp = client.chat.completions.create(
        model=AGENT_MODEL,
        messages=[
            {"role": "system", "content": CHAT_RESPONSE_SYSTEM},
            {"role": "user", "content":
                f"User message:\n{user_message}\n\n"
                f"Retrieved evidence:\n{chunk_text}\n\n"
                f"Top trial IDs: {[t.get('nct_id') for t in retrieved.get('top_trials', [])]}"},
        ],
        temperature=0.2,
    )
    answer = resp.choices[0].message.content.strip()
    return {
        "answer": answer,
        "search_query": search_query,
        "retrieval": retrieved,
    }


def evaluate_trial(patient: dict, nct_id: str, title: str, chunks: list[dict]) -> dict:
    chunk_text = "\n\n---\n\n".join(
        f"CHUNK_ID: {c['id']}\n{c['text']}" for c in chunks
    )
    resp = client.chat.completions.create(
        model=AGENT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EVAL_SYSTEM},
            {"role": "user", "content":
                f"Patient:\n{json.dumps(patient, indent=2)}\n\n"
                f"Trial: {title} ({nct_id})\n\n"
                f"Retrieved criteria chunks — cite the CHUNK_ID when using a chunk:\n\n"
                f"{chunk_text}"},
        ],
        temperature=0, # set temp to 0 for deterministic output 
    )
    result = json.loads(resp.choices[0].message.content)

    # Validate cited_chunk_ids are real
    valid_ids = {c["id"] for c in chunks}
    for check in result.get("criteria_checks", []):
        cited = check.get("cited_chunk_id")
        if cited and cited not in valid_ids:
            check["cited_chunk_id"] = None  # reject invalid citations

    return result

def run_agent(patient: dict, verbose: bool = True) -> dict:
    """
    Full pipeline: retrieve → per-trial eligibility check → re-retrieve if needed.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Patient: {patient.get('patient_id', patient.get('name', '?'))}")
        print(f"Diagnosis: {patient.get('primary_diagnosis')}")

    # ── Retrieval ────────────────────────────────────────────────────────────
    expanded_patient = {
    **patient,
    "primary_diagnosis": expand_diagnosis(patient.get("primary_diagnosis", "")),
    }
    ret = retrieve(expanded_patient, verbose=verbose)

    # Group fused chunks by trial
    trial_chunks: dict[str, list[dict]] = defaultdict(list)
    for chunk in ret["fused_results"]:
        trial_chunks[chunk["nct_id"]].append(chunk)

    if verbose:
        print(f"  Candidate trials: {list(trial_chunks.keys())}")

    # ── Per-trial evaluation with re-retrieval ───────────────────────────────
    verdicts = []
    retrieval_log = []

    for nct_id, chunks in trial_chunks.items():
        title   = chunks[0]["metadata"].get("title", nct_id)
        attempt = 0

        while attempt <= MAX_RETRIES:
            verdict = evaluate_trial(patient, nct_id, title, chunks)
            conf    = verdict.get("confidence", 1.0)

            log_entry = {
                "nct_id":   nct_id,
                "attempt":  attempt + 1,
                "confidence": conf,
                "verdict":  verdict.get("verdict"),
            }
            
            missing = verdict.get("missing_information", [])
            needs_retry = conf < CONFIDENCE_THRESH 
            if not needs_retry or attempt >= MAX_RETRIES:
                verdicts.append(verdict)
                log_entry["accepted"] = True
                retrieval_log.append(log_entry)
                if verbose:
                    flag = "" if conf >= CONFIDENCE_THRESH else " ⚠ low-conf, max retries"
                    print(f"  [{verdict['verdict'].upper()}] {nct_id} "
                          f"(conf={conf:.2f}{flag}, attempt {attempt+1})")
                break
            else:
                # Re-retrieve: rewrite query focused on missing info
                missing = verdict.get("missing_information", [])
                rewrite_prompt = (
                    f"Original query about: {patient.get('primary_diagnosis')}\n"
                    f"Missing: {missing}\nTrial: {title}"
                )
                rw_resp = client.chat.completions.create(
                    model=AGENT_MODEL,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": REWRITE_SYSTEM},
                        {"role": "user",   "content": rewrite_prompt},
                    ],
                    temperature=0,
                )
                new_query = json.loads(rw_resp.choices[0].message.content)["rewritten_query"]
                new_emb   = embed(new_query)
                extra     = retrieve_ann(new_emb, top_k=5, chunk_type="exclusion")
                # Merge new chunks (avoid duplicates)
                existing_ids = {c["id"] for c in chunks}
                chunks += [c for c in extra if c["id"] not in existing_ids]

                log_entry["rewritten_query"] = new_query
                retrieval_log.append(log_entry)
                if verbose:
                    print(f"  [{nct_id}] conf={conf:.2f} < threshold, "
                          f"re-retrieving (attempt {attempt+2})...")
                attempt += 1

    # Sort: eligible → uncertain → not_eligible
    order = {"eligible": 0, "uncertain": 1, "not_eligible": 2}
    verdicts.sort(key=lambda v: order.get(v.get("verdict", "uncertain"), 1))

    summary = {
        "total_evaluated": len(verdicts),
        "eligible":        sum(1 for v in verdicts if v["verdict"] == "eligible"),
        "uncertain":       sum(1 for v in verdicts if v["verdict"] == "uncertain"),
        "not_eligible":    sum(1 for v in verdicts if v["verdict"] == "not_eligible"),
        "re_retrievals":   sum(1 for e in retrieval_log if e.get("attempt", 1) > 1),
    }

    if verbose:
        s = summary
        print(f"\nResults: {s['eligible']} eligible | "
              f"{s['uncertain']} uncertain | {s['not_eligible']} not eligible")
        print(f"Re-retrievals: {s['re_retrievals']}")

    return {
        "patient":         patient,
        "retrieval":       ret,
        "retrieval_log":   retrieval_log,
        "verdicts":        verdicts,
        "summary":         summary,
    }


# ── Demo ──────────────────────────────────────────────────────────────────────
# ── Structured extraction ─────────────────────────────────────────────────────
EXTRACTION_SYSTEM = """Extract structured patient fields from the user message.
Return JSON only, no preamble:
{
  "conditions": [],
  "healthy_volunteer": null,
  "age": null,
  "sex": null,
  "medications": [],
  "prior_treatments": [],
  "lab_values": {},
  "location": null,
  "preferences": {
    "remote_only": false,
    "max_travel_miles": null,
    "paid_only": false
  }
}

Rules:
- If the person mentions no health conditions and says they are healthy, 
  set healthy_volunteer to true and leave conditions empty.
- If they mention specific conditions, populate conditions normally.
- Return null for any field not mentioned."""


def extract_patient_fields(user_message: str, conversation_context: str = "") -> dict:
    resp = client.chat.completions.create(
        model=AGENT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content":
                f"Conversation so far:\n{conversation_context}\n\nLatest message:\n{user_message}"},
        ],
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)

DEMO_PATIENT = {
    "patient_id":        "DEMO_001",
    "age":               58,
    "sex":               "Male",
    "primary_diagnosis": "Type 2 diabetes mellitus",
    "lab_values": {
        "HbA1c":           "8.9%",
        "eGFR":            "62 mL/min/1.73m²",
        "fasting_glucose": "187 mg/dL",
        "BMI":             "31.2 kg/m²",
    },
    "current_medications": ["Metformin 1000mg BID", "Lisinopril 10mg daily"],
    "prior_treatments":    ["Metformin (current)", "Glipizide (discontinued — hypoglycemia)"],
    "comorbidities":       ["Hypertension", "Mild CKD stage 2"],
    "exclusion_flags": {
        "insulin_use": False, "recent_MI": False,
        "active_malignancy": False, "pregnancy": False,
    },
}

if __name__ == "__main__":
    result = run_agent(DEMO_PATIENT, verbose=True)
    os.makedirs("data", exist_ok=True)
    with open("data/agent_result_demo.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("\nSaved to data/agent_result_demo.json")
