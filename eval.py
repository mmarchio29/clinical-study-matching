"""
eval.py — Full evaluation suite.

Two-stage evaluation that treats retrieval and generation separately:

STAGE 1 — RETRIEVAL EVALUATION (does the system find the right trials?)
  - recall@k:   of the relevant trials, how many appear in top-k?
  - MRR:        mean reciprocal rank of first relevant result
  - NDCG@k:     normalised discounted cumulative gain (graded relevance)

  Ablation comparisons (same patients, different retriever configs):
    A. HyDE + ANN + BM25 + RRF  (full pipeline)
    B. Direct embed + ANN only  (no HyDE, no BM25)
    C. BM25 only                (keyword baseline)
    D. HNSW ef sweep            (recall vs latency tradeoff)

STAGE 2 — GENERATION EVALUATION (does the agent make correct decisions?)
  - Verdict accuracy / precision / recall / F1 vs ground truth
  - Citation grounding rate
  - Hallucination rate (baseline LLM citing fake NCT IDs)
  - Re-retrieval loop impact

Run:
    python ingest.py                   # build indexes first
    python eval.py --build-gt          # auto-build silver ground truth
    python eval.py                     # run full eval
"""

import argparse
import json
import math
import os
import pickle
import time
from collections import defaultdict

import chromadb
from openai import OpenAI

from retriever import (
    CHROMA_PATH, COLLECTION_NAME, BM25_PATH,
    EMBED_MODEL, AGENT_MODEL, TOP_K, FINAL_K, RRF_K,
    embed, retrieve, retrieve_ann, retrieve_bm25,
    reciprocal_rank_fusion, run_agent, DEMO_PATIENT,
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GT_PATH       = "./data/ground_truth_retrieval.json"
PATIENTS_PATH = "./data/synthetic_patients.json"
RESULTS_DIR   = "./data/eval_results"
BENCHMARK_PATH = "./data/benchmark_reviewed.json"


# ══════════════════════════════════════════════════════════════════════════════
# Ground truth construction
# ══════════════════════════════════════════════════════════════════════════════

BUILD_GT_SYSTEM = """You are a clinical trial eligibility expert.
Given a patient profile and a trial's eligibility criteria, decide
if this trial is RELEVANT for this patient (worth screening them for).

Relevance = the patient plausibly meets enough criteria to warrant
a closer look. This is softer than full eligibility.

Return ONLY JSON: {"relevant": true | false, "reason": "one sentence"}"""


def build_ground_truth(patients: list[dict], trials_per_patient: int = 8) -> dict:
    """
    For each patient, retrieve a broad candidate set then use GPT-4o
    to label each (patient, trial) pair as relevant or not.
    Saves silver labels to GT_PATH — review before final eval.
    """
    chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(COLLECTION_NAME)

    gt = {}  # {patient_id: {nct_id: true/false}}

    for i, patient in enumerate(patients):
        pid = patient["patient_id"]
        print(f"  Labelling patient {pid} ({i+1}/{len(patients)})...")

        # Broad retrieval to get candidate NCT IDs
        summary = (
            f"{patient['primary_diagnosis']} "
            f"HbA1c {patient['lab_values'].get('HbA1c','')} "
            f"eGFR {patient['lab_values'].get('eGFR','')}"
        )
        emb     = embed(summary)
        results = collection.query(
            query_embeddings=[emb],
            n_results=trials_per_patient * 3,
            where={"type": "inclusion"},
            include=["documents", "metadatas"],
        )

        # Deduplicate to unique NCT IDs
        seen, candidates = set(), []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            nct = meta.get("nct_id")
            if nct and nct not in seen:
                seen.add(nct)
                candidates.append({"nct_id": nct, "text": doc})
            if len(candidates) >= trials_per_patient:
                break

        patient_gt = {}
        for cand in candidates:
            resp = client.chat.completions.create(
                model=AGENT_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": BUILD_GT_SYSTEM},
                    {"role": "user", "content":
                        f"Patient:\n{json.dumps(patient, indent=2)}\n\n"
                        f"Trial {cand['nct_id']} criteria:\n{cand['text']}"},
                ],
                temperature=0,
            )
            label = json.loads(resp.choices[0].message.content)
            patient_gt[cand["nct_id"]] = label["relevant"]

        gt[pid] = patient_gt

    os.makedirs("data", exist_ok=True)
    with open(GT_PATH, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"  Silver labels saved to {GT_PATH}")
    print("  IMPORTANT: review and correct before final evaluation.")
    return gt


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Retrieval metrics
# ══════════════════════════════════════════════════════════════════════════════

def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant trials found in top-k retrieved."""
    if not relevant_ids:
        return 0.0
    hits = sum(1 for r in retrieved_ids[:k] if r in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """
    MRR = mean of 1/rank of the first relevant result.
    Rewards systems that put a relevant result at rank 1 vs rank 5.
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """
    NDCG@k: Normalised Discounted Cumulative Gain.
    Rewards relevant results appearing earlier in the ranking.
    Uses binary relevance (1 if relevant, 0 if not).
    """
    def dcg(ids):
        return sum(
            (1 if ids[i] in relevant_ids else 0) / math.log2(i + 2)
            for i in range(min(k, len(ids)))
        )
    ideal_hits = min(k, len(relevant_ids))
    ideal = dcg(list(relevant_ids)[:ideal_hits])
    if ideal == 0:
        return 0.0
    return dcg(retrieved_ids[:k]) / ideal


def eval_retriever_config(
    patients: list[dict],
    ground_truth: dict,
    config: dict,
    k_values: list[int] = [1, 3, 5, 10],
) -> dict:
    """
    Evaluate one retriever configuration across all patients.

    config keys:
        use_hyde (bool)
        use_bm25 (bool)
        label (str)
    """
    label     = config.get("label", "unnamed")
    use_hyde  = config.get("use_hyde", True)
    use_bm25  = config.get("use_bm25", True)

    metrics_per_k = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}
    latencies = []

    for patient in patients:
        pid = patient["patient_id"]
        if pid not in ground_truth:
            continue

        relevant_ncts = {nct for nct, rel in ground_truth[pid].items() if rel}
        if not relevant_ncts:
            continue

        # Run retrieval with this config
        t0  = time.perf_counter()
        ret = retrieve(
            patient,
            top_k=max(k_values),
            final_k=max(k_values),
            use_hyde=use_hyde,
            use_bm25=use_bm25,
            verbose=False,
        )
        latencies.append(time.perf_counter() - t0)

        # Extract unique NCT IDs in ranked order from fused results
        retrieved_ncts = list(dict.fromkeys(
            c["nct_id"] for c in ret["fused_results"]
        ))

        for k in k_values:
            metrics_per_k[k]["recall"].append(recall_at_k(retrieved_ncts, relevant_ncts, k))
            metrics_per_k[k]["mrr"].append(mean_reciprocal_rank(retrieved_ncts, relevant_ncts))
            metrics_per_k[k]["ndcg"].append(ndcg_at_k(retrieved_ncts, relevant_ncts, k))

    summary = {"label": label, "n_patients": len(latencies),
               "avg_latency_s": round(sum(latencies) / max(len(latencies), 1), 3)}
    for k in k_values:
        m = metrics_per_k[k]
        n = len(m["recall"])
        if n == 0:
            continue
        summary[f"recall@{k}"] = round(sum(m["recall"]) / n, 3)
        summary[f"mrr@{k}"]    = round(sum(m["mrr"])    / n, 3)
        summary[f"ndcg@{k}"]   = round(sum(m["ndcg"])   / n, 3)

    return summary


def run_ablation(patients: list[dict], ground_truth: dict) -> list[dict]:
    """
    Ablation study: compare 3 retriever configurations to isolate
    the contribution of each technique.

    A = full pipeline (HyDE + ANN + BM25 + RRF)
    B = no HyDE, ANN only          — isolates HyDE's contribution
    C = BM25 only                  — shows what keyword search gets
    D = no BM25, ANN + HyDE only   — isolates BM25's contribution
    """
    configs = [
        {"label": "A: HyDE+ANN+BM25+RRF", "use_hyde": True,  "use_bm25": True},
        {"label": "B: direct+ANN only",    "use_hyde": False, "use_bm25": False},
        {"label": "C: BM25 only",          "use_hyde": False, "use_bm25": True},
        {"label": "D: HyDE+ANN only",      "use_hyde": True,  "use_bm25": False},
    ]
    results = []
    for cfg in configs:
        print(f"  Running config: {cfg['label']}")
        r = eval_retriever_config(patients, ground_truth, cfg)
        results.append(r)
        print(f"    recall@5={r.get('recall@5','—')}  ndcg@5={r.get('ndcg@5','—')}  MRR={r.get('mrr@5','—')}")
    return results

# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Generation metrics
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_SYSTEM = """You are a clinical research specialist.
Given a patient profile, list recruiting clinical trials the patient may qualify for.
Return JSON:
{
  "trials": [
    {"nct_id": "...", "trial_title": "...", "verdict": "eligible"|"not_eligible"|"uncertain",
     "reasoning": "..."}
  ]
}"""


def run_baseline(patient: dict) -> dict:
    """Vanilla GPT-4o, no retrieval."""
    resp = client.chat.completions.create(
        model=AGENT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM},
            {"role": "user",   "content":
                f"Patient:\n{json.dumps(patient, indent=2)}\n\n"
                f"List recruiting trials for {patient.get('primary_diagnosis','this condition')}."},
        ],
        temperature=0,
    )
    result = json.loads(resp.choices[0].message.content)
    result["patient"] = patient
    return result


def verdict_metrics(agent_results: list[dict], ground_truth_verdicts: dict) -> dict:
    """Precision/recall/F1 for the 'eligible' class."""
    tp = fp = fn = tn = 0
    for result in agent_results:
        pid = result["patient"]["patient_id"]
        if pid not in ground_truth_verdicts:
            continue
        for verdict in result.get("verdicts", []):
            nct = verdict.get("nct_id")
            gt  = ground_truth_verdicts[pid].get(nct)
            if gt is None:
                continue
            pred_pos = verdict.get("verdict") == "eligible"
            true_pos = gt is True
            if pred_pos and true_pos:  tp += 1
            if pred_pos and not true_pos: fp += 1
            if not pred_pos and true_pos: fn += 1
            if not pred_pos and not true_pos: tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def paid_local_feasibility_metrics(agent_results: list[dict], k: int = 5) -> dict:
    """
    Measures business-critical relevance:
      - feasibility_precision@k
      - paid_match_precision@k
      - local_match_precision@k
    """
    total = feasible = paid = local = 0
    for result in agent_results:
        patient = result.get("patient", {})
        prefs = patient.get("preferences", {})
        min_comp = prefs.get("minimum_compensation")
        user_loc = patient.get("location", {})
        user_city = str(user_loc.get("city", "")).lower()
        user_state = str(user_loc.get("state", "")).lower()
        user_country = str(user_loc.get("country", "")).lower()

        top_trials = result.get("retrieval", {}).get("top_trials", [])[:k]
        for t in top_trials:
            total += 1
            m = t.get("metadata", {})
            trial_paid = (m.get("amount_min") is not None) or (m.get("amount_max") is not None)
            trial_local = (
                (user_city and str(m.get("site_city", "")).lower() == user_city) or
                (user_state and str(m.get("site_state", "")).lower() == user_state) or
                (user_country and str(m.get("site_country", "")).lower() == user_country) or
                bool(m.get("remote_allowed", False))
            )
            comp_ok = True
            if min_comp is not None:
                observed = m.get("amount_max") if m.get("amount_max") is not None else m.get("amount_min")
                comp_ok = (observed is not None and observed >= min_comp)

            if trial_paid:
                paid += 1
            if trial_local:
                local += 1
            if trial_paid and trial_local and comp_ok:
                feasible += 1

    denom = max(total, 1)
    return {
        f"n_trials@{k}": total,
        f"feasibility_precision@{k}": round(feasible / denom, 3),
        f"paid_match_precision@{k}": round(paid / denom, 3),
        f"local_match_precision@{k}": round(local / denom, 3),
    }


def load_benchmark_labels(default_gt: dict) -> dict:
    """
    Prefer reviewed benchmark labels; fall back to silver labels.
    """
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded reviewed benchmark labels from {BENCHMARK_PATH}")
        return data
    print(f"No reviewed benchmark at {BENCHMARK_PATH}; using silver labels from {GT_PATH}")
    return default_gt


def hallucination_rate(baseline_results: list[dict], valid_ncts: set[str]) -> dict:
    """What fraction of NCT IDs cited by the baseline don't exist in our corpus?"""
    total = hallucinated = 0
    for result in baseline_results:
        for trial in result.get("trials", []):
            nct = trial.get("nct_id", "")
            total += 1
            valid_fmt = nct.startswith("NCT") and len(nct) == 11 and nct[3:].isdigit()
            if not valid_fmt or nct not in valid_ncts:
                hallucinated += 1
    return {
        "total_citations": total,
        "hallucinated":    hallucinated,
        "rate":            round(hallucinated / total, 3) if total > 0 else 0.0,
    }


def citation_grounding(agent_results: list[dict]) -> dict:
    """What fraction of criterion checks cite a real chunk ID?"""
    total = grounded = 0
    for result in agent_results:
        for verdict in result.get("verdicts", []):
            for check in verdict.get("criteria_checks", []):
                total += 1
                if check.get("cited_chunk_id") not in (None, "null", ""):
                    grounded += 1
    return {
        "total_checks": total,
        "grounded":     grounded,
        "rate":         round(grounded / total, 3) if total > 0 else 0.0,
    }


def retrieval_loop_stats(agent_results: list[dict]) -> dict:
    total = re_retrieved = 0
    for result in agent_results:
        for entry in result.get("retrieval_log", []):
            total += 1
            if entry.get("attempt", 1) > 1:
                re_retrieved += 1
    return {
        "total_evaluations": total,
        "re_retrievals":     re_retrieved,
        "rate":              round(re_retrieved / total, 3) if total > 0 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def load_patients(n: int = 20) -> list[dict]:
    if not os.path.exists(PATIENTS_PATH):
        print(f"No patients file found at {PATIENTS_PATH}.")
        print("Using demo patient only.")
        return [DEMO_PATIENT]
    with open(PATIENTS_PATH) as f:
        return json.load(f)[:n]


def main(build_gt: bool = False, n_patients: int = 20):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    patients = load_patients(n_patients)

    # Get valid NCT IDs from corpus
    chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(COLLECTION_NAME)
    valid_ncts = {m.get("nct_id") for m in collection.get(include=["metadatas"])["metadatas"]}
    print(f"Corpus: {len(valid_ncts)} unique trials, {len(patients)} patients")

    # Ground truth
    if build_gt or not os.path.exists(GT_PATH):
        print("\nBuilding ground truth labels...")
        ground_truth = build_ground_truth(patients[:15])
    else:
        with open(GT_PATH) as f:
            ground_truth = json.load(f)
        print(f"Loaded ground truth: {len(ground_truth)} patients labelled")
    benchmark_gt = load_benchmark_labels(ground_truth)

    # ── Stage 1: Retrieval evaluation ────────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 1 — RETRIEVAL EVALUATION")
    print("="*60)

    print("\n[Ablation study]")
    ablation = run_ablation(patients, ground_truth)

    # ── Stage 2: Generation evaluation ───────────────────────────────────────
    print("\n" + "="*60)
    print("STAGE 2 — GENERATION EVALUATION")
    print("="*60)

    print(f"\nRunning agent on {len(patients)} patients...")
    agent_results = []
    for i, p in enumerate(patients):
        print(f"  [{i+1}/{len(patients)}] {p['patient_id']}")
        try:
            agent_results.append(run_agent(p, verbose=False))
        except Exception as e:
            print(f"    Error: {e}")

    print(f"\nRunning baseline on {len(patients)} patients...")
    baseline_results = []
    for i, p in enumerate(patients):
        print(f"  [{i+1}/{len(patients)}] {p['patient_id']}")
        try:
            baseline_results.append(run_baseline(p))
        except Exception as e:
            print(f"    Error: {e}")

    vm   = verdict_metrics(agent_results, benchmark_gt)
    bvm  = verdict_metrics(baseline_results, ground_truth)
    hall = hallucination_rate(baseline_results, valid_ncts)
    cit  = citation_grounding(agent_results)
    loop = retrieval_loop_stats(agent_results)
    feas = paid_local_feasibility_metrics(agent_results, k=5)

    # ── Print summary report ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    print("\n── Retrieval ablation (recall@5 / NDCG@5 / MRR) ──")
    for r in ablation:
        print(f"  {r['label']:<30}  "
              f"recall@5={r.get('recall@5','—'):<6}  "
              f"ndcg@5={r.get('ndcg@5','—'):<6}  "
              f"mrr={r.get('mrr@5','—')}")


    print("\n── Generation: verdict accuracy ──")
    print(f"  Agent:    P={vm['precision']}  R={vm['recall']}  F1={vm['f1']}")
    print(f"  Baseline: P={bvm['precision']}  R={bvm['recall']}  F1={bvm['f1']}")

    print(f"\n── Hallucination (baseline) ──")
    print(f"  {hall['hallucinated']}/{hall['total_citations']} cited NCT IDs not in corpus  "
          f"({hall['rate']:.0%} hallucination rate)")

    print(f"\n── Citation grounding (agent) ──")
    print(f"  {cit['grounded']}/{cit['total_checks']} criteria checks cite a real chunk  "
          f"({cit['rate']:.0%} grounding rate)")

    print(f"\n── Re-retrieval loop ──")
    print(f"  {loop['re_retrievals']}/{loop['total_evaluations']} evaluations triggered re-retrieval  "
          f"({loop['rate']:.0%})")

    print("\n── Paid/local feasibility ──")
    print(f"  feasibility@5={feas['feasibility_precision@5']}  "
          f"paid@5={feas['paid_match_precision@5']}  "
          f"local@5={feas['local_match_precision@5']}")

    # Save full report
    report = {
        "ablation":           ablation,
        "verdict_agent":      vm,
        "verdict_baseline":   bvm,
        "hallucination":      hall,
        "citation_grounding": cit,
        "retrieval_loop":     loop,
        "paid_local_feasibility": feas,
    }
    out = f"{RESULTS_DIR}/eval_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {out}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-gt",    action="store_true", help="Rebuild ground truth labels")
    parser.add_argument("--n-patients",  type=int, default=20)
    args = parser.parse_args()
    main(build_gt=args.build_gt, n_patients=args.n_patients)
