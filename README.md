# Clinical Study Matching

A RAG-powered clinical trial matching system that helps patients find relevant recruiting studies in Boston, MA. The pipeline combines semantic search, keyword retrieval, and an LLM eligibility agent to surface the trials most likely to match a patient's profile.

---

## How It Works

The system uses four layered retrieval techniques:

**1. HyDE (Hypothetical Document Embeddings)** — Rather than embedding the patient profile directly, GPT-4o generates what the ideal matching trial's eligibility criteria would look like. That hypothetical document is embedded instead. This bridges the vocabulary gap between patient language ("I have anxiety") and clinical trial language ("DSM-5 confirmed GAD").

**2. ANN via HNSW** — ChromaDB's approximate nearest neighbor index (using HNSW graph traversal) retrieves semantically similar trial chunks from the embedded corpus.

**3. BM25** — A parallel keyword search over the same corpus. BM25 catches exact term matches that semantic search might miss, acting as a hard-term-matching complement.

**4. RRF (Reciprocal Rank Fusion)** — Results from ANN and BM25 are merged without needing to normalize scores across different spaces. Documents appearing in both lists get an additive boost.

After retrieval, a GPT-4o eligibility agent evaluates each candidate trial criterion-by-criterion, with an automatic re-retrieval loop for low-confidence verdicts.

---

## Project Structure

```
clinical-study-matching/
├── ingest.py          # Fetch trials, chunk, embed, and index into ChromaDB + BM25
├── retriever.py       # Full RAG pipeline: HyDE → ANN + BM25 → RRF → agent loop
├── chat_intake.py     # Conversational interface: collects patient info, runs agent
├── eval.py            # Evaluation suite: retrieval metrics + generation metrics
├── requirements.txt
└── data/
    ├── chroma_db/             # ChromaDB vector index
    ├── bm25_index.pkl         # Serialized BM25 index
    ├── chunks.json            # All chunked trial text
    ├── trials_raw.json        # Raw parsed trial records
    ├── ground_truth_retrieval.json   # Silver labels for retrieval eval
    ├── synthetic_patients.json       # Synthetic patient profiles for eval
    └── eval_results/
        └── eval_report.json   # Full evaluation report
```

---

## Setup

### Prerequisites

- Python 3.10+
- An OpenAI API key

### Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
openai
chromadb
python-dotenv
rank_bm25
requests
tiktoken
```

### Environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...

# Data source: "paid_file" (default) or "ctgov" (ClinicalTrials.gov API)
TRIAL_SOURCE_TYPE=paid_file

# Path to your paid trial source JSON (if using paid_file)
PAID_SOURCE_PATH=./data/paid_trials_source.json

# Location filter (used for ctgov source)
TRIAL_CITY=Boston
TRIAL_STATE=MA

# Max trials to fetch/index
MAX_TRIALS=3000
```

---

## Usage

### Step 1 — Build the index

Run once to fetch trials, chunk them, embed with OpenAI, and store in ChromaDB and BM25:

```bash
python ingest.py
```

This creates `data/chroma_db/`, `data/bm25_index.pkl`, and `data/chunks.json`.

#### Data sources

Set `TRIAL_SOURCE_TYPE` in your `.env`:

- `paid_file` (default) — reads from a local JSON file at `PAID_SOURCE_PATH`. The file should be a JSON array of trial records with fields like `nct_id`, `title`, `eligibility_criteria`, `comp_text`, etc.
- `ctgov` — fetches live recruiting trials from the ClinicalTrials.gov v2 API filtered to Boston, MA.

### Step 2 — Run the chat interface

```bash
python chat_intake.py
```

The assistant collects your age, sex, and health conditions conversationally, then runs the full retrieval and eligibility pipeline and presents matching trials.

### Step 3 — Run a patient directly (demo)

```bash
python retriever.py
```

Runs the demo patient (a 58-year-old male with Type 2 diabetes) through the full pipeline and saves results to `data/agent_result_demo.json`.

---

## Evaluation

The eval suite measures both retrieval quality and generation quality.

### Generate synthetic patients

```bash
python eval.py --generate-patients --n-patients 20
```

### Build ground truth labels (calls GPT-4o per patient × trial)

```bash
python eval.py --build-gt
```

Review and correct `data/ground_truth_retrieval.json`, then optionally save reviewed labels to `data/benchmark_reviewed.json`.

### Run the full evaluation

```bash
python eval.py
```

**Stage 1 — Retrieval metrics** (recall@k, MRR, NDCG@k) across four ablation configs:

| Config | Description |
|--------|-------------|
| A | HyDE + ANN + BM25 + RRF (full pipeline) |
| B | Direct embed + ANN only (no HyDE, no BM25) |
| C | BM25 only (keyword baseline) |
| D | HyDE + ANN only (no BM25) |

**Stage 2 — Generation metrics:**
- Verdict precision / recall / F1 vs ground truth
- Hallucination rate (no-RAG baseline vs RAG pipeline)
- Citation grounding rate (how often criteria checks cite a real retrieved chunk)
- Re-retrieval loop rate and uncertain-to-definitive resolution rate

Results are saved to `data/eval_results/eval_report.json`.

---

## Architecture Notes

### Chunking strategy

Each trial produces three chunk types stored separately in ChromaDB:
- `inclusion` — the inclusion criteria text
- `exclusion` — the exclusion criteria text
- `metadata` — trial overview (conditions, interventions, age range, phases, compensation, location)

Keeping chunk types separate allows HyDE queries to target the right semantic space.

### HNSW parameters

Configured in `ingest.py`:

```python
HNSW_EF_CONSTRUCTION = 200  # Higher = better graph quality, slower build
HNSW_M               = 32   # Edges per node — higher = better recall, more memory
```

`ef_search` (query-time recall/speed tradeoff) is set in `retriever.py`:

```python
EF_SEARCH = 100
```

### Agent re-retrieval loop

When the eligibility agent returns `confidence < 0.75` for a trial, it rewrites the retrieval query focused on the missing information and fetches additional exclusion chunks. This repeats up to 2 times before accepting the verdict.

---

## Notes

- All trials are assumed to be in Boston, MA — location is not evaluated as an eligibility criterion.
- Self-reported patient data is treated as unconfirmed; the agent marks criteria as `uncertain` rather than `does_not_meet` when clinical documentation is absent.
- The system never guarantees eligibility — results should be verified with a physician or study coordinator.
