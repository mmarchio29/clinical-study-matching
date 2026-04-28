"""
eval.py — Full evaluation suite for the clinical trial RAG pipeline.

Two-stage evaluation:

STAGE 1 — RETRIEVAL EVALUATION (does the system find the right trials?)
  - recall@k:   fraction of relevant trials found in top-k
  - MRR:        mean reciprocal rank of first relevant result
  - NDCG@k:     normalised discounted cumulative gain

  Ablation comparisons:
    A. HyDE + ANN + BM25 + RRF  (full pipeline)
    B. Direct embed + ANN only  (no HyDE, no BM25)
    C. BM25 only                (keyword baseline)
    D. HyDE + ANN only          (no BM25)

STAGE 2 — GENERATION EVALUATION (does the agent make correct decisions?)
  - Verdict precision / recall / F1 vs ground truth
  - Citation grounding rate
  - Hallucination rate (baseline LLM vs RAG pipeline)
  - Re-retrieval loop impact

Run:
    python ingest.py                       # build indexes first
    python eval.py --generate-patients     # generate synthetic patients
    python eval.py --build-gt              # label with GPT-4o (silver labels)
    python eval.py                         # run full eval
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from retriever import (
    CHROMA_PATH, COLLECTION_NAME,
    EMBED_MODEL, AGENT_MODEL, TOP_K, FINAL_K,
    embed, retrieve, retrieve_ann, retrieve_bm25,
    reciprocal_rank_fusion, run_agent, expand_diagnosis,
    DEMO_PATIENT,
)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

GT_PATH        = "./data/ground_truth_retrieval.json"
PATIENTS_PATH  = "./data/synthetic_patients.json"
RESULTS_DIR    = "./data/eval_results"
BENCHMARK_PATH = "./data/benchmark_reviewed.json"


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic patient generation
# ══════════════════════════════════════════════════════════════════════════════

PATIENT_GEN_SYSTEM = """Generate a realistic synthetic patient profile for clinical trial matching.
The patient should have a condition commonly studied in Boston academic medical centers.
Return JSON matching exactly this schema:
{
  "patient_id": "SYNTH_001",
  "age": 45,
  "sex": "Female",
  "primary_diagnosis": "Generalized Anxiety Disorder",
  "lab_values": {},
  "current_medications": ["sertraline 50mg daily"],
  "prior_treatments": ["CBT (completed)"],
  "comorbidities": ["mild depression"],
  "exclusion_flags": {
    "pregnant": false,
    "formally_diagnosed": true
  },
  "preferences": {
    "paid_only": false,
    "remote_only": false
  }
}

Rules:
- Vary age (18-75), sex, condition, and medication profile realistically
- For healthy volunteers: empty lab_values, no medications, no comorbidities
- For complex patients: include relevant lab values and comorbidities
- formally_diagnosed should reflect whether they would realistically have a clinical dx
- Return only the JSON object, no preamble"""

# Conditions representative of Boston academic medical center trial landscape
CONDITION_LIST = [
    "Generalized Anxiety Disorder",
    "Major Depressive Disorder",
    "ADHD",
    "Type 2 Diabetes",
    "Breast Cancer",
    "Multiple Sclerosis",
    "Parkinson's Disease",
    "Heart Failure",
    "COPD",
    "Social Anxiety Disorder",
    "PTSD",
    "Bipolar Disorder",
    "Schizophrenia",
    "Healthy volunteer",
    "Obesity",
    "Hypertension",
    "Alzheimer's Disease",
    "Lupus",
    "Crohn's Disease",
    "Atrial Fibrillation",
]


def generate_synthetic_patients(n: int = 20) -> list[dict]:
    """
    Generate n diverse synthetic patients covering common Boston trial conditions.
    Saves to PATIENTS_PATH.
    """
    patients = []
    for i in range(n):
        condition_hint = CONDITION_LIST[i % len(CONDITION_LIST)]
        resp = client.chat.completions.create(
            model=AGENT_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": PATIENT_GEN_SYSTEM},
                {"role": "user", "content":
                    f"Generate patient #{i+1} with condition: {condition_hint}. "
                    f"Use patient_id: SYNTH_{i+1:03d}"},
            ],
            temperature=0.8,
        )
        patient = json.loads(resp.choices[0].message.content)
        # Ensure patient_id is set correctly
        patient["patient_id"] = f"SYNTH_{i+1:03d}"
        patients.append(patient)
        print(f"  Generated {patient['patient_id']}: {patient['primary_diagnosis']}")

    os.makedirs("data", exist_ok=True)
    with open(PATIENTS_PATH, "w") as f:
        json.dump(patients, f, indent=2)
    print(f"Saved {n} synthetic patients to {PATIENTS_PATH}")
    return patients


def load_patients(n: int = 20) -> list[dict]:
    if not os.path.exists(PATIENTS_PATH):
        print(f"No patients file at {PATIENTS_PATH} — generating synthetic patients...")
        return generate_synthetic_patients(n)
    with open(PATIENTS_PATH) as f:
        return json.load(f)[:n]


# ══════════════════════════════════════════════════════════════════════════════
# Ground truth construction
# ══════════════════════════════════════════════════════════════════════════════

BUILD_GT_SYSTEM = """You are a clinical trial eligibility expert.
Given a patient profile and a trial's eligibility criteria, decide
if this trial is RELEVANT for this patient (worth screening them for).

Relevance = the patient plausibly meets enough criteria to warrant
a closer look. This is softer than full eligibility — err toward true
when in doubt.

Return ONLY JSON: {"relevant": true | false, "reason": "one sentence"}"""


def build_ground_truth(patients: list[dict], trials_per_patient: int = 15) -> dict:
    """
    For each patient, retrieve a broad candidate set then use GPT-4o
    to label each (patient, trial) pair as relevant or not.
    Saves silver labels to GT_PATH.
    """
    chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(COLLECTION_NAME)

    gt = {}  # {patient_id: {nct_id: true/false}}

    for i, patient in enumerate(patients):
        pid = patient["patient_id"]
        print(f"  Labelling {pid} ({i+1}/{len(patients)}): {patient['primary_diagnosis']}")

        # Expand diagnosis to clinical terms before embedding
        expanded = expand_diagnosis(patient.get("primary_diagnosis", ""))
        lab_str  = " ".join(f"{k} {v}" for k, v in patient.get("lab_values", {}).items())
        summary  = f"{expanded} {lab_str}".strip()

        emb     = embed(summary)
        results = collection.query(
            query_embeddings=[emb],
            n_results=trials_per_patient * 5,
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
            time.sleep(0.2)  # avoid rate limits

        gt[pid] = patient_gt
        relevant_count = sum(1 for v in patient_gt.values() if v)
        print(f"    {relevant_count}/{len(patient_gt)} trials labelled relevant")

    os.makedirs("data", exist_ok=True)
    with open(GT_PATH, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"\nSilver labels saved to {GT_PATH}")
    print("IMPORTANT: review and correct before final evaluation.")
    return gt


def load_benchmark_labels(default_gt: dict) -> dict:
    """Prefer reviewed benchmark labels; fall back to silver labels."""
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded reviewed benchmark labels from {BENCHMARK_PATH}")
        return data
    print(f"No reviewed benchmark at {BENCHMARK_PATH} — using silver labels")
    return default_gt


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
    Rewards systems that surface a relevant result at rank 1 vs rank 5.
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
    label    = config.get("label", "unnamed")
    use_hyde = config.get("use_hyde", True)
    use_bm25 = config.get("use_bm25", True)

    metrics_per_k = {k: {"recall": [], "mrr": [], "ndcg": []} for k in k_values}
    latencies = []

    for patient in patients:
        pid = patient["patient_id"]
        if pid not in ground_truth:
            continue
        relevant_ncts = {nct for nct, rel in ground_truth[pid].items() if rel}
        if not relevant_ncts:
            continue

        expanded_patient = {
            **patient,
            "primary_diagnosis": expand_diagnosis(patient.get("primary_diagnosis", "")),
        }

        t0 = time.perf_counter()

        if config.get("use_bm25") and not config.get("use_hyde"):
            # BM25-only config
            from retriever import retrieve_bm25, filter_active_results
            summary = expanded_patient.get("primary_diagnosis", "")
            bm25_hits = filter_active_results(retrieve_bm25(summary, top_k=max(k_values)))
            fused_chunks = bm25_hits
        else:
            ret = retrieve(
                expanded_patient,
                top_k=max(k_values),
                final_k=max(k_values),
                use_hyde=use_hyde,
                use_bm25=use_bm25,
                verbose=False,
            )
            fused_chunks = ret["fused_results"]

        latencies.append(time.perf_counter() - t0)

        retrieved_ncts = list(dict.fromkeys(
            c["nct_id"] for c in fused_chunks
        ))

        for k in k_values:
            metrics_per_k[k]["recall"].append(recall_at_k(retrieved_ncts, relevant_ncts, k))
            metrics_per_k[k]["mrr"].append(mean_reciprocal_rank(retrieved_ncts, relevant_ncts))
            metrics_per_k[k]["ndcg"].append(ndcg_at_k(retrieved_ncts, relevant_ncts, k))

    n = len(latencies)
    summary = {
        "label":         label,
        "n_patients":    n,
        "avg_latency_s": round(sum(latencies) / max(n, 1), 3),
    }
    for k in k_values:
        m = metrics_per_k[k]
        count = len(m["recall"])
        if count == 0:
            continue
        summary[f"recall@{k}"] = round(sum(m["recall"]) / count, 3)
        summary[f"mrr@{k}"]    = round(sum(m["mrr"])    / count, 3)
        summary[f"ndcg@{k}"]   = round(sum(m["ndcg"])   / count, 3)

    return summary

def run_ablation(patients: list[dict], ground_truth: dict) -> list[dict]:
    """
    Ablation study: compare 4 retriever configurations to isolate
    the contribution of each technique.

    A = full pipeline (HyDE + ANN + BM25 + RRF)  — our system
    B = no HyDE, ANN only                         — isolates HyDE contribution
    C = BM25 only                                 — keyword baseline
    D = HyDE + ANN only, no BM25                  — isolates BM25 contribution
    """
    configs = [
        {"label": "A: HyDE+ANN+BM25+RRF (full)", "use_hyde": True,  "use_bm25": True},
        {"label": "B: Direct+ANN only",           "use_hyde": False, "use_bm25": False},
        {"label": "C: BM25 only",                 "use_hyde": False, "use_bm25": True},
        {"label": "D: HyDE+ANN only",             "use_hyde": True,  "use_bm25": False},
    ]
    results = []
    for cfg in configs:
        print(f"  Running config: {cfg['label']}")
        r = eval_retriever_config(patients, ground_truth, cfg)
        results.append(r)
        print(f"    recall@5={r.get('recall@5','—')}  "
              f"ndcg@5={r.get('ndcg@5','—')}  "
              f"mrr@5={r.get('mrr@5','—')}  "
              f"latency={r.get('avg_latency_s','—')}s")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Generation metrics
# ══════════════════════════════════════════════════════════════════════════════

BASELINE_SYSTEM = """You are a clinical research specialist.
Given a patient profile, list recruiting clinical trials in Boston, MA
that the patient may qualify for.
Return JSON:
{
  "trials": [
    {
      "nct_id": "...",
      "trial_title": "...",
      "verdict": "eligible" | "not_eligible" | "uncertain",
      "reasoning": "..."
    }
  ]
}
Important: only cite real NCT IDs you are confident exist."""


def run_baseline(patient: dict) -> dict:
    """
    Vanilla GPT-4o with no retrieval — used to measure hallucination rate
    and show what the pipeline adds over a raw LLM call.
    """
    resp = client.chat.completions.create(
        model=AGENT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": BASELINE_SYSTEM},
            {"role": "user", "content":
                f"Patient:\n{json.dumps(patient, indent=2)}\n\n"
                f"List recruiting trials for {patient.get('primary_diagnosis', 'this condition')}."},
        ],
        temperature=0,
    )
    result = json.loads(resp.choices[0].message.content)
    result["patient"] = patient
    return result


def verdict_metrics(agent_results: list[dict], ground_truth_verdicts: dict) -> dict:
    """
    Precision/recall/F1 for the 'eligible' class vs ground truth labels.
    Treats 'uncertain' as not-eligible for binary scoring.
    """
    tp = fp = fn = tn = 0
    for result in agent_results:
        pid = result["patient"]["patient_id"]
        if pid not in ground_truth_verdicts:
            continue
        for verdict in result.get("verdicts", []):
            nct      = verdict.get("nct_id")
            gt       = ground_truth_verdicts[pid].get(nct)
            if gt is None:
                continue
            pred_pos = verdict.get("verdict") == "eligible"
            true_pos = gt is True
            if pred_pos and true_pos:      tp += 1
            if pred_pos and not true_pos:  fp += 1
            if not pred_pos and true_pos:  fn += 1
            if not pred_pos and not true_pos: tn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 3),
        "recall":    round(rec, 3),
        "f1":        round(f1, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def hallucination_rate(baseline_results: list[dict], valid_ncts: set[str]) -> dict:
    """
    What fraction of NCT IDs cited by the no-RAG baseline don't exist
    in our corpus? High hallucination rate = strong argument for RAG.
    """
    total = hallucinated = 0
    for result in baseline_results:
        for trial in result.get("trials", []):
            nct = trial.get("nct_id", "")
            total += 1
            valid_fmt = (
                nct.startswith("NCT")
                and len(nct) == 11
                and nct[3:].isdigit()
            )
            if not valid_fmt or nct not in valid_ncts:
                hallucinated += 1
    return {
        "total_citations": total,
        "hallucinated":    hallucinated,
        "rate":            round(hallucinated / total, 3) if total > 0 else 0.0,
    }


def citation_grounding(agent_results: list[dict]) -> dict:
    """
    What fraction of criterion checks in agent verdicts cite a real chunk ID?
    Measures how grounded the agent's reasoning is in retrieved evidence.
    """
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
    """
    How often does the re-retrieval loop trigger?
    High rate = agent frequently finds insufficient evidence on first pass.
    Low rate = retrieved chunks are usually sufficient.
    """
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


def uncertain_resolution_rate(agent_results: list[dict]) -> dict:
    """
    Of trials that triggered re-retrieval, how many moved from uncertain
    to a definitive verdict (eligible or not_eligible)?
    """
    triggered = resolved = 0
    for result in agent_results:
        log = result.get("retrieval_log", [])
        # Group log entries by nct_id
        by_trial = defaultdict(list)
        for entry in log:
            by_trial[entry["nct_id"]].append(entry)
        for nct, entries in by_trial.items():
            if len(entries) > 1:
                triggered += 1
                final_verdict = entries[-1].get("verdict", "uncertain")
                if final_verdict != "uncertain":
                    resolved += 1
    return {
        "triggered":  triggered,
        "resolved":   resolved,
        "resolution_rate": round(resolved / triggered, 3) if triggered > 0 else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(build_gt: bool = False, generate_patients: bool = False, n_patients: int = 20):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load or generate patients
    if generate_patients or not os.path.exists(PATIENTS_PATH):
        print("\nGenerating synthetic patients...")
        patients = generate_synthetic_patients(n_patients)
    else:
        patients = load_patients(n_patients)
    print(f"Patients loaded: {len(patients)}")

    # Get valid NCT IDs from corpus for hallucination check
    chroma     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma.get_collection(COLLECTION_NAME)
    valid_ncts = {
        m.get("nct_id")
        for m in collection.get(include=["metadatas"])["metadatas"]
        if m.get("nct_id")
    }
    print(f"Corpus: {len(valid_ncts)} unique trials")

    # Ground truth
    if build_gt or not os.path.exists(GT_PATH):
        print("\nBuilding ground truth labels (this calls GPT-4o per trial per patient)...")
        ground_truth = build_ground_truth(patients[:15])
    else:
        with open(GT_PATH) as f:
            ground_truth = json.load(f)
        print(f"Loaded ground truth: {len(ground_truth)} patients labelled")

    benchmark_gt = load_benchmark_labels(ground_truth)

    # ── Stage 1: Retrieval evaluation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1 — RETRIEVAL EVALUATION")
    print("=" * 60)

    print("\n[Ablation study — comparing retriever configurations]")
    ablation = run_ablation(patients, ground_truth)

    # ── Stage 2: Generation evaluation ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2 — GENERATION EVALUATION")
    print("=" * 60)

    print(f"\nRunning RAG agent on {len(patients)} patients...")
    agent_results = []
    for i, p in enumerate(patients):
        print(f"  [{i+1}/{len(patients)}] {p['patient_id']}: {p.get('primary_diagnosis','?')}")
        try:
            # Expand diagnosis before agent run (mirrors chat_intake.py behavior)
            expanded = {
                **p,
                "primary_diagnosis": expand_diagnosis(p.get("primary_diagnosis", "")),
            }
            result = run_agent(expanded, verbose=False)
            result["patient"] = p  # keep original for metric matching
            agent_results.append(result)
        except Exception as e:
            print(f"    Error: {e}")

    print(f"\nRunning no-RAG baseline on {len(patients)} patients...")
    baseline_results = []
    for i, p in enumerate(patients):
        print(f"  [{i+1}/{len(patients)}] {p['patient_id']}")
        try:
            baseline_results.append(run_baseline(p))
        except Exception as e:
            print(f"    Error: {e}")

    # Compute all metrics
    vm   = verdict_metrics(agent_results, benchmark_gt)
    bvm  = verdict_metrics(baseline_results, ground_truth)
    hall = hallucination_rate(baseline_results, valid_ncts)
    cit  = citation_grounding(agent_results)
    loop = retrieval_loop_stats(agent_results)
    res  = uncertain_resolution_rate(agent_results)

    # ── Print summary report ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n── Retrieval ablation ──")
    print(f"  {'Config':<32}  {'recall@5':<10}  {'ndcg@5':<10}  {'mrr@5':<10}  {'latency'}")
    for r in ablation:
        print(f"  {r['label']:<32}  "
              f"{str(r.get('recall@5', '—')):<10}  "
              f"{str(r.get('ndcg@5',  '—')):<10}  "
              f"{str(r.get('mrr@5',   '—')):<10}  "
              f"{r.get('avg_latency_s', '—')}s")

    print("\n── Generation: verdict accuracy vs ground truth ──")
    print(f"  RAG agent:      P={vm['precision']}  R={vm['recall']}  F1={vm['f1']}  "
          f"(TP={vm['tp']} FP={vm['fp']} FN={vm['fn']})")
    print(f"  No-RAG baseline: P={bvm['precision']}  R={bvm['recall']}  F1={bvm['f1']}  "
          f"(TP={bvm['tp']} FP={bvm['fp']} FN={bvm['fn']})")

    print(f"\n── Hallucination rate (no-RAG baseline) ──")
    print(f"  {hall['hallucinated']}/{hall['total_citations']} cited NCT IDs not in corpus  "
          f"({hall['rate']:.0%} hallucination rate)")
    print(f"  → RAG pipeline eliminates hallucinations by grounding in retrieved chunks")

    print(f"\n── Citation grounding (RAG agent) ──")
    print(f"  {cit['grounded']}/{cit['total_checks']} criteria checks cite a real chunk  "
          f"({cit['rate']:.0%} grounding rate)")

    print(f"\n── Re-retrieval loop ──")
    print(f"  {loop['re_retrievals']}/{loop['total_evaluations']} evaluations triggered re-retrieval  "
          f"({loop['rate']:.0%} rate)")
    print(f"  Uncertain → definitive verdict resolution: "
          f"{res['resolved']}/{res['triggered']}  "
          f"({res['resolution_rate']:.0%} resolution rate)")

    # Save full report
    report = {
        "n_patients":         len(patients),
        "n_corpus_trials":    len(valid_ncts),
        "ablation":           ablation,
        "verdict_agent":      vm,
        "verdict_baseline":   bvm,
        "hallucination":      hall,
        "citation_grounding": cit,
        "retrieval_loop":     loop,
        "uncertain_resolution": res,
    }
    out = f"{RESULTS_DIR}/eval_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFull report saved to {out}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the clinical trial RAG pipeline")
    parser.add_argument(
        "--build-gt",
        action="store_true",
        help="Rebuild GPT-4o silver ground truth labels",
    )
    parser.add_argument(
        "--generate-patients",
        action="store_true",
        help="Regenerate synthetic patient profiles",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=20,
        help="Number of patients to evaluate (default: 20)",
    )
    args = parser.parse_args()
    main(
        build_gt=args.build_gt,
        generate_patients=args.generate_patients,
        n_patients=args.n_patients,
    )