# Intelligent Candidate Discovery Ranking System

Track 1 submission project for the Redrob / India runs Data & AI challenge.

This repository contains a local, CPU-only candidate ranker for the Senior AI Engineer job description. It streams the provided `candidates.jsonl`, scores each profile with explainable feature buckets, and writes the top 100 candidates in the official submission format.

## What It Builds

- `outputs/submission.csv` - official validator-ready top 100 ranking.
- `outputs/ranked_candidates.xlsx` - same ranking in XLSX form for portals that request a spreadsheet upload.
- `docs/methodology.md` and `docs/methodology.pdf` - explanation of the approach.

## Setup

Use Python 3.10 or newer. No package installation is required for ranking.

Place the challenge dataset here:

```text
data/candidates.jsonl
```

The full candidate file is not committed because it is large. Copy it from the organizer bundle before running.

## Run

```bash
python rank.py --candidates data/candidates.jsonl --out outputs/submission.csv --xlsx outputs/ranked_candidates.xlsx
```

Expected runtime on the 100,000-candidate dataset is well under the 5-minute CPU-only constraint because the code streams JSONL and keeps only the current top 100 in memory.

## Validate

If you have the organizer validator in the repo root:

```bash
python validate_submission.py outputs/submission.csv data/candidates.jsonl
```

## Ranking Method

The system combines:

- Role fit: senior AI/ML/search/ranking engineering titles and 5-9 years of experience.
- Production evidence: shipped retrieval, ranking, semantic search, vector search, recommendation systems, NDCG/MRR/MAP, A/B testing, and recruiter/candidate matching systems.
- Skills: Python, NLP, embeddings, vector databases, BM25, semantic search, learning-to-rank, LLM fine-tuning, and supporting ML tooling.
- Career context: product company, AI/ML, marketplace, startup, and hands-on engineering environments.
- Behavioral availability: recent activity, recruiter response rate, open-to-work flag, notice period, interview completion, and verification signals.
- Risk penalties: non-engineering keyword stuffing, recent course-only AI exposure, consulting-only background, stale profiles, long notice periods, and honeypot-like inconsistencies.

Each row includes a short reasoning sentence tied to candidate facts used by the score.

## Reproducibility

The single command above produces the ranked CSV from the raw dataset. The ranker does not use network access, GPUs, hosted LLM APIs, or manual edits.
