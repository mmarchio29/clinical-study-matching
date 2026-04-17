# Clinical Study Matching

A RAG-based pipeline that matches patients to recruiting clinical trials using a hybrid retrieval system and an LLM-powered eligibility agent.

Built on top of the [ClinicalTrials.gov API](https://clinicaltrials.gov/api/v2/studies), ChromaDB, OpenAI, and BM25 — with a full evaluation suite that measures both retrieval quality and generation correctness.

---

## How It Works

The system is split into three stages:

**1. Ingestion (`ingest.py`)**  
Fetches recruiting trials from ClinicalTrials.gov, parses and chunks each trial into three document types (inclusion criteria, exclusion criteria, and a metadata overview), embeds them with OpenAI's `text-embedding-3-small`, and stores them in a ChromaDB HNSW index. A BM25 index is built in parallel over the same corpus.

**2. Retrieval + Agent (`retriever.py`)**  
Given a patient profile, the pipeline runs four techniques in sequence:

- **HyDE (Hypothetical Document Embeddings)** — Instead of embedding the patient profile directly, GPT-4o generates what a perfectly matching trial's eligibility criteria would look like. That hypothetical document is embedded, bridging the vocabulary gap between patient records and trial criteria.
- **ANN via HNSW** — ChromaDB performs approximate nearest-neighbor search using the HyDE embedding. The `ef_search` parameter controls the recall/speed tradeoff.
- **BM25** — Keyword search over the same corpus runs in parallel, catching exact clinical term matches that semantic search may miss.
- **Reciprocal Rank Fusion (RRF)** — Merges the ANN and BM25 ranked lists into a single ranking without needing to normalize scores across the two spaces.

The top fused candidates are then passed to a GPT-4o eligibility agent, which evaluates each trial criterion-by-criterion. If confidence falls below a threshold, the agent rewrites its query and re-retrieves before rendering a final verdict.

**3. Evaluation (`eval.py`)**  
A two-stage evaluation suite:

- **Stage 1 (Retrieval)** — Measures recall@k, MRR, and NDCG@k across four ablation configurations (full pipeline vs. no HyDE, BM25 only, ANN only) and sweeps HNSW `ef_search` to plot recall vs. latency.
- **Stage 2 (Generation)** — Measures verdict precision/recall/F1 against ground truth labels, citation grounding rate, hallucination rate on a retrieval-free baseline, and re-retrieval loop impact.

---

## Project Structure

```
.
├── ingest.py         # Fetch, parse, chunk, embed, index
├── retriever.py      # HyDE + ANN + BM25 + RRF + agent loop
├── eval.py           # Retrieval & generation evaluation suite
└── data/
    ├── chroma_db/                    # ChromaDB HNSW vector store
    ├── bm25_index.pkl                # Serialized BM25 index
    ├── chunks.json                   # All chunked trial documents
    ├── trials_raw.json               # Raw API responses
    ├── synthetic_patients.json       # Patient cohort for evaluation
    ├── ground_truth_retrieval.json   # Silver relevance labels
    └── eval_results/
        └── eval_report.json          # Full evaluation output
```

---

## Setup

**Requirements:** Python 3.11+, an OpenAI API key.

```bash
pip install openai chromadb rank-bm25 requests
export OPENAI_API_KEY="sk-..."
```

---

## Usage

**Step 1 — Build the indexes** (fetches up to 300 recruiting T2D trials):

```bash
python ingest.py
```

**Step 2 — Run on the demo patient:**

```bash
python retriever.py
```

**Step 3 — Build silver ground truth labels and run full evaluation:**

```bash
python eval.py --build-gt     # label (patient, trial) pairs with GPT-4o
python eval.py                # run full eval
```

The evaluation report is saved to `data/eval_results/eval_report.json`.

---

## Configuration

Key parameters are set at the top of each file:

| Parameter | File | Default | Description |
|---|---|---|---|
| `CONDITION` | `ingest.py` | `"type 2 diabetes"` | Condition to query from ClinicalTrials.gov |
| `MAX_TRIALS` | `ingest.py` | `300` | Maximum trials to fetch |
| `HNSW_EF_CONSTRUCTION` | `ingest.py` | `200` | Graph quality at index build time |
| `HNSW_M` | `ingest.py` | `32` | Edges per node in the HNSW graph |
| `TOP_K` | `retriever.py` | `10` | Candidates per retriever before fusion |
| `FINAL_K` | `retriever.py` | `5` | Trials passed to the agent after RRF |
| `EF_SEARCH` | `retriever.py` | `100` | HNSW nodes explored at query time |
| `CONFIDENCE_THRESH` | `retriever.py` | `0.55` | Confidence below which re-retrieval triggers |

---

## Retrieval Ablations

The eval suite compares four retriever configurations to isolate the contribution of each technique:

| Config | HyDE | ANN | BM25 |
|---|---|---|---|
| A: Full pipeline | ✓ | ✓ | ✓ |
| B: Direct embed + ANN only | — | ✓ | — |
| C: BM25 only | — | — | ✓ |
| D: HyDE + ANN only | ✓ | ✓ | — |

The `ef_search` sweep runs configs with `ef` from 10 to 300 to characterize the ANN recall/latency tradeoff.

---

## Models

| Role | Model |
|---|---|
| Embeddings | `text-embedding-3-small` |
| HyDE generation, agent, ground truth labeling | `gpt-4o` |
