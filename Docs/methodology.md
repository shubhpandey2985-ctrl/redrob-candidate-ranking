# Methodology: Intelligent Candidate Discovery

## Objective

Rank the top 100 professionals for Redrob AI's Senior AI Engineer role. The job description asks for a practical builder who has shipped production retrieval, ranking, matching, and evaluation systems, not simply someone with AI keywords in a skills list.

## Architecture

The system is a deterministic, CPU-only ranking pipeline:

1. Stream `candidates.jsonl` one record at a time.
2. Build a compact evidence text from the profile, career history, industries, titles, and skills.
3. Score each candidate across explainable buckets.
4. Keep only the top 100 candidates in a heap for memory efficiency.
5. Emit the official CSV plus an XLSX copy with the same columns.

No hosted LLM APIs, GPU inference, or network calls are used during ranking.

## Scoring Buckets

### 1. Role And Seniority Fit

Candidates receive strong positive weight for AI, ML, NLP, search, ranking, applied scientist, backend, data, and senior engineering titles. The experience curve favors roughly 5-9 years, with the strongest region around 6-8 years, matching the JD's stated intent.

### 2. Production Retrieval And Ranking Evidence

The highest-weight signals come from career descriptions, not from isolated skills. The system rewards evidence such as production ranking pipelines, hybrid retrieval, semantic search, BM25 plus dense recall, vector search, NDCG/MRR/MAP, A/B tests, recruiter-facing search, recommendation systems, candidate-JD matching, and operational scale.

### 3. Skills Inventory

The ranker scores skills based on relevance, proficiency, duration, and endorsements. High-value skills include Python, NLP, information retrieval, embeddings, semantic search, vector databases, FAISS, Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, BM25, learning-to-rank, recommendation systems, and LLM fine-tuning.

### 4. Career Context

The JD explicitly values product-building judgment. The system therefore boosts candidates with product-company, AI/ML, marketplace, fintech, SaaS, e-commerce, or startup exposure. It penalizes consulting-only backgrounds when there is no product engineering evidence.

### 5. Behavioral Availability

Behavioral signals adjust the profile fit score. Recent activity, open-to-work status, recruiter response rate, short notice period, interview completion rate, profile completeness, recruiter saves, and GitHub activity all improve confidence that the candidate is hireable now.

### 6. Risk And Trap Handling

The dataset contains keyword stuffing and honeypot-style profiles. The system down-weights non-engineering titles that merely mention AI keywords, course-only or side-project-only AI exposure, stale profiles, long notice periods, CV/speech/robotics-only profiles without IR/NLP/ranking evidence, and implausible skill-duration patterns.

## Explainability

Every output row includes a 1-2 sentence reason using facts from the candidate profile: current title, years of experience, location, matched skills, production evidence, response rate, notice period, and any relevant concern.

## Compute Fit

The implementation streams JSONL and keeps a 100-item heap, so memory usage is small relative to the 16 GB limit. The ranker uses only Python's standard library and avoids network calls, GPU dependencies, or per-candidate LLM calls.

## Expected Strength

The approach is designed to avoid the main challenge trap: ranking candidates by keyword overlap alone. Career-history evidence and behavioral availability have meaningful weight, while isolated skills or recent AI curiosity do not dominate the score.
