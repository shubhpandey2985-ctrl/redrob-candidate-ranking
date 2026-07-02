#!/usr/bin/env python3
"""Generate a ranked Redrob Track 1 candidate submission.

The ranker is deliberately local, deterministic, and explainable. It streams the
candidate JSONL file, extracts role-specific evidence, applies transparent
scoring buckets, then writes the top 100 candidates in the official submission
format. No network, GPU, or hosted LLM calls are used during ranking.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable


AS_OF_DATE = date(2026, 6, 1)

CONSULTING_COMPANIES = {
    "tcs",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "mindtree",
    "ltimindtree",
    "hcl",
    "tech mahindra",
    "genpact",
}

PRODUCT_INDUSTRIES = {
    "software",
    "ai/ml",
    "fintech",
    "e-commerce",
    "saas",
    "adtech",
    "healthtech",
    "healthtech ai",
    "conversational ai",
    "gaming",
    "transportation",
    "insurance tech",
    "internet",
    "media",
    "food delivery",
}

PREFERRED_CITIES = {
    "pune": 7.0,
    "noida": 7.0,
    "gurgaon": 5.0,
    "delhi": 5.0,
    "bangalore": 5.0,
    "bengaluru": 5.0,
    "hyderabad": 5.0,
    "mumbai": 5.0,
    "chennai": 3.0,
    "kolkata": 2.0,
}

TECH_TITLE_TERMS = {
    "ai",
    "machine learning",
    "ml",
    "nlp",
    "applied scientist",
    "data scientist",
    "search",
    "ranking",
    "software engineer",
    "backend engineer",
    "data engineer",
    "analytics engineer",
    "cloud engineer",
    "devops engineer",
    "full stack",
}

NON_TECH_TITLE_TERMS = {
    "marketing",
    "sales",
    "hr",
    "accountant",
    "customer support",
    "graphic designer",
    "mechanical",
    "civil",
    "operations manager",
    "content writer",
    "business analyst",
    "project manager",
}

CORE_SKILLS = {
    "python": 6.0,
    "machine learning": 5.0,
    "deep learning": 3.0,
    "nlp": 5.0,
    "information retrieval": 8.0,
    "semantic search": 8.0,
    "vector search": 8.0,
    "embeddings": 8.0,
    "sentence transformers": 7.0,
    "bge": 6.0,
    "e5": 5.0,
    "bm25": 7.0,
    "learning to rank": 8.0,
    "ranking systems": 8.0,
    "recommendation systems": 6.0,
    "llms": 4.0,
    "rag": 5.0,
    "fine-tuning llms": 4.0,
    "lora": 3.0,
    "qlora": 3.0,
    "peft": 3.0,
    "pytorch": 4.0,
    "tensorflow": 3.0,
    "scikit-learn": 3.0,
}

VECTOR_DB_SKILLS = {
    "faiss": 8.0,
    "pinecone": 7.0,
    "weaviate": 7.0,
    "qdrant": 7.0,
    "milvus": 7.0,
    "opensearch": 7.0,
    "elasticsearch": 6.0,
    "pgvector": 6.0,
    "haystack": 4.0,
    "llamaindex": 2.0,
}

PRODUCTION_PHRASES = {
    "production": 8.0,
    "shipped": 8.0,
    "deployed": 6.0,
    "serving": 6.0,
    "queries per month": 7.0,
    "p95": 4.0,
    "a/b": 7.0,
    "ab test": 7.0,
    "online evaluation": 7.0,
    "offline": 4.0,
    "ndcg": 9.0,
    "mrr": 7.0,
    "map": 7.0,
    "relevance": 5.0,
    "labeling pipeline": 5.0,
    "evaluation harness": 8.0,
    "ranking pipeline": 9.0,
    "learning-to-rank": 9.0,
    "hybrid retrieval": 9.0,
    "semantic search": 8.0,
    "embedding-based": 8.0,
    "dense retrieval": 8.0,
    "bm25": 7.0,
    "candidate-jd": 9.0,
    "recruiter-facing": 8.0,
    "recommendation system": 7.0,
    "marketplace": 4.0,
    "behavioral-signal": 5.0,
    "index refresh": 4.0,
    "retrieval-quality": 5.0,
}

TRAP_PHRASES = {
    "online courses": -12.0,
    "taking online courses": -14.0,
    "side projects": -9.0,
    "curious about how ai": -12.0,
    "augment my work": -10.0,
    "openai api": -5.0,
    "langchain": -3.0,
    "demo": -4.0,
    "tutorial": -5.0,
    "research-only": -20.0,
    "pure research": -18.0,
    "academic lab": -16.0,
}

CV_SPEECH_ROBOTICS = {
    "computer vision",
    "image classification",
    "opencv",
    "yolo",
    "speech recognition",
    "tts",
    "asr",
    "robotics",
    "gans",
    "diffusion models",
}


@dataclass(order=True)
class RankedCandidate:
    sort_key: tuple[float, str] = field(init=False, repr=False)
    candidate_id: str
    raw_score: float
    reasoning: str
    diagnostics: dict[str, float]

    def __post_init__(self) -> None:
        self.sort_key = (-self.raw_score, self.candidate_id)


def norm_text(value: Any) -> str:
    return str(value or "").lower()


def contains_any(text: str, needles: Iterable[str]) -> bool:
    return any(needle in text for needle in needles)


def parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def phrase_score(text: str, weights: dict[str, float]) -> tuple[float, list[str]]:
    score = 0.0
    hits: list[str] = []
    for phrase, weight in weights.items():
        if phrase in text:
            score += weight
            hits.append(phrase)
    return score, hits


def skill_score(skills: list[dict[str, Any]]) -> tuple[float, list[str], dict[str, dict[str, Any]]]:
    by_name: dict[str, dict[str, Any]] = {}
    total = 0.0
    hits: list[str] = []
    prof_mult = {"beginner": 0.35, "intermediate": 0.7, "advanced": 1.0, "expert": 1.15}

    for skill in skills:
        name = norm_text(skill.get("name"))
        by_name[name] = skill
        base = CORE_SKILLS.get(name, 0.0) + VECTOR_DB_SKILLS.get(name, 0.0)
        if not base:
            continue
        months = float(skill.get("duration_months") or 0)
        duration_mult = clamp(months / 36.0, 0.35, 1.2)
        endorsements = math.log1p(float(skill.get("endorsements") or 0)) / 4.5
        total += base * prof_mult.get(norm_text(skill.get("proficiency")), 0.7) * duration_mult
        total += min(2.0, endorsements)
        hits.append(skill.get("name", name))
    return min(total, 75.0), hits, by_name


def title_score(title: str) -> float:
    score = 0.0
    if contains_any(title, TECH_TITLE_TERMS):
        score += 18.0
    if any(seniority in title for seniority in ["senior", "lead", "staff", "principal"]):
        score += 8.0
    if contains_any(title, NON_TECH_TITLE_TERMS):
        score -= 35.0
    return score


def experience_score(years: float) -> float:
    if 5.0 <= years <= 9.0:
        return 20.0 - abs(years - 7.0) * 1.8
    if 4.0 <= years < 5.0 or 9.0 < years <= 10.5:
        return 8.0
    if years > 13.0:
        return -10.0
    return -16.0


def location_score(profile: dict[str, Any], signals: dict[str, Any]) -> float:
    location = norm_text(profile.get("location"))
    country = norm_text(profile.get("country"))
    score = 0.0
    for city, value in PREFERRED_CITIES.items():
        if city in location:
            score += value
            break
    if country != "india":
        score -= 7.0
    if signals.get("willing_to_relocate"):
        score += 3.0
    if signals.get("preferred_work_mode") in {"hybrid", "flexible"}:
        score += 2.0
    return score


def behavior_score(signals: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    notes: list[str] = []
    last_active = parse_date(signals.get("last_active_date"))
    if last_active:
        days_inactive = (AS_OF_DATE - last_active).days
        if days_inactive <= 14:
            score += 8.0
            notes.append("recently active")
        elif days_inactive <= 45:
            score += 4.0
        elif days_inactive > 120:
            score -= 10.0
            notes.append("stale activity")

    response_rate = float(signals.get("recruiter_response_rate") or 0)
    score += (response_rate - 0.45) * 18.0
    if response_rate >= 0.7:
        notes.append(f"{response_rate:.0%} recruiter response")
    elif response_rate < 0.2:
        notes.append(f"low {response_rate:.0%} recruiter response")

    avg_hours = float(signals.get("avg_response_time_hours") or 0)
    if avg_hours <= 48:
        score += 3.0
    elif avg_hours > 168:
        score -= 4.0

    notice = int(signals.get("notice_period_days") or 0)
    if notice <= 30:
        score += 6.0
        notes.append(f"{notice}-day notice")
    elif notice >= 90:
        score -= 8.0
        notes.append(f"{notice}-day notice")

    score += clamp((float(signals.get("profile_completeness_score") or 0) - 70.0) / 5.0, -5.0, 6.0)
    score += min(5.0, math.log1p(float(signals.get("saved_by_recruiters_30d") or 0)))
    score += min(4.0, math.log1p(float(signals.get("profile_views_received_30d") or 0)) / 1.3)
    score += min(5.0, max(0.0, float(signals.get("github_activity_score") or 0)) / 20.0)
    score += (float(signals.get("interview_completion_rate") or 0) - 0.6) * 8.0

    if signals.get("open_to_work_flag"):
        score += 5.0
        notes.append("open to work")
    if signals.get("verified_email") and signals.get("verified_phone"):
        score += 2.0
    return score, notes


def career_context_score(candidate: dict[str, Any], text: str) -> tuple[float, list[str]]:
    history = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    score = 0.0
    notes: list[str] = []
    industries = {norm_text(j.get("industry")) for j in history}
    companies = {norm_text(j.get("company")) for j in history}
    current_industry = norm_text(profile.get("current_industry"))

    if current_industry in PRODUCT_INDUSTRIES or industries & PRODUCT_INDUSTRIES:
        score += 9.0
        notes.append("product/AI industry exposure")
    if all((ind in {"it services", "consulting"} or comp in CONSULTING_COMPANIES) for ind, comp in zip(industries or {""}, companies or {""})):
        score -= 18.0
        notes.append("consulting-heavy background")
    if "startup" in text or "series a" in text or "founding" in text:
        score += 4.0
        notes.append("startup/founding context")
    if "mentoring" in text or "led the team" in text or "owned" in text:
        score += 4.0
    return score, notes


def honeypot_penalty(candidate: dict[str, Any], skill_map: dict[str, dict[str, Any]], text: str) -> tuple[float, list[str]]:
    penalty = 0.0
    notes: list[str] = []
    expert_zero = sum(
        1
        for skill in skill_map.values()
        if norm_text(skill.get("proficiency")) == "expert" and int(skill.get("duration_months") or 0) <= 3
    )
    if expert_zero >= 3:
        penalty -= 18.0
        notes.append("implausible expert skill durations")

    profile_years = float(candidate.get("profile", {}).get("years_of_experience") or 0)
    total_months = sum(int(j.get("duration_months") or 0) for j in candidate.get("career_history", []))
    if profile_years and total_months > (profile_years * 12.0 + 36.0):
        penalty -= 8.0
        notes.append("career duration mismatch")

    if contains_any(text, {"expert in 10", "0 years used", "founded 3 years ago"}):
        penalty -= 25.0
        notes.append("honeypot-like wording")
    return penalty, notes


def build_candidate_text(candidate: dict[str, Any]) -> str:
    profile = candidate.get("profile", {})
    history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    parts = [
        profile.get("headline", ""),
        profile.get("summary", ""),
        profile.get("current_title", ""),
        profile.get("current_industry", ""),
    ]
    parts.extend(j.get("title", "") for j in history)
    parts.extend(j.get("industry", "") for j in history)
    parts.extend(j.get("description", "") for j in history)
    parts.extend(s.get("name", "") for s in skills)
    return norm_text(" ".join(parts))


def score_candidate(candidate: dict[str, Any]) -> RankedCandidate:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    text = build_candidate_text(candidate)
    title = norm_text(profile.get("current_title"))
    years = float(profile.get("years_of_experience") or 0)

    diagnostics: dict[str, float] = {}
    notes: list[str] = []

    diagnostics["experience"] = experience_score(years)
    diagnostics["title"] = title_score(title)

    skills_total, skill_hits, skill_map = skill_score(candidate.get("skills", []))
    diagnostics["skills"] = skills_total
    if skill_hits:
        notes.append("skills: " + ", ".join(skill_hits[:5]))

    prod_total, prod_hits = phrase_score(text, PRODUCTION_PHRASES)
    diagnostics["production"] = min(prod_total, 85.0)
    if prod_hits:
        notes.append("evidence: " + ", ".join(prod_hits[:4]))

    trap_total, trap_hits = phrase_score(text, TRAP_PHRASES)
    diagnostics["trap_penalty"] = trap_total
    if trap_hits:
        notes.append("risk: " + ", ".join(trap_hits[:2]))

    context_total, context_notes = career_context_score(candidate, text)
    diagnostics["career_context"] = context_total
    notes.extend(context_notes)

    diagnostics["location"] = location_score(profile, signals)

    behavior_total, behavior_notes = behavior_score(signals)
    diagnostics["behavior"] = behavior_total
    notes.extend(behavior_notes)

    hp_total, hp_notes = honeypot_penalty(candidate, skill_map, text)
    diagnostics["honeypot_penalty"] = hp_total
    notes.extend(hp_notes)

    if contains_any(title, NON_TECH_TITLE_TERMS) and (skill_hits or prod_hits):
        diagnostics["keyword_stuffer_penalty"] = -28.0
        notes.append("non-engineering title despite AI keywords")
    else:
        diagnostics["keyword_stuffer_penalty"] = 0.0

    if contains_any(text, CV_SPEECH_ROBOTICS) and not contains_any(text, {"retrieval", "ranking", "search", "nlp", "recommendation"}):
        diagnostics["domain_mismatch_penalty"] = -12.0
        notes.append("CV/speech-heavy without IR evidence")
    else:
        diagnostics["domain_mismatch_penalty"] = 0.0

    raw = sum(diagnostics.values())
    reasoning = make_reasoning(candidate, diagnostics, notes, skill_hits, prod_hits)
    return RankedCandidate(candidate.get("candidate_id", ""), raw, reasoning, diagnostics)


def make_reasoning(
    candidate: dict[str, Any],
    diagnostics: dict[str, float],
    notes: list[str],
    skill_hits: list[str],
    prod_hits: list[str],
) -> str:
    profile = candidate.get("profile", {})
    signals = candidate.get("redrob_signals", {})
    title = profile.get("current_title", "Candidate")
    years = float(profile.get("years_of_experience") or 0)
    location = profile.get("location", "unknown location")
    response = float(signals.get("recruiter_response_rate") or 0)
    notice = int(signals.get("notice_period_days") or 0)

    strengths: list[str] = []
    if prod_hits:
        strengths.append("production evidence around " + ", ".join(prod_hits[:3]))
    if skill_hits:
        strengths.append("matched skills include " + ", ".join(skill_hits[:4]))
    if diagnostics.get("career_context", 0) > 0:
        strengths.append("product/AI career context")
    if not strengths:
        strengths.append("reasonable adjacent engineering background")

    concerns: list[str] = []
    if diagnostics.get("trap_penalty", 0) < -8 or diagnostics.get("keyword_stuffer_penalty", 0) < 0:
        concerns.append("screened for keyword-only risk")
    if notice >= 60:
        concerns.append(f"{notice}-day notice")
    if response < 0.3:
        concerns.append(f"low {response:.0%} recruiter response")
    if norm_text(profile.get("country")) != "india":
        concerns.append("outside India")

    first = f"{title} with {years:.1f} years in {location}; {strengths[0]}."
    if concerns:
        second = "Concern: " + "; ".join(concerns[:2]) + "."
    else:
        second = f"Behavioral fit is solid with {response:.0%} recruiter response and {notice}-day notice."
    return (first + " " + second)[:900]


def iter_candidates(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def select_top(candidates_path: Path, limit: int = 100) -> list[RankedCandidate]:
    import heapq

    heap: list[tuple[float, str, RankedCandidate]] = []
    for candidate in iter_candidates(candidates_path):
        scored = score_candidate(candidate)
        item = (scored.raw_score, scored.candidate_id, scored)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda x: (-x[0], x[1]))]


def normalized_scores(rows: list[RankedCandidate]) -> list[float]:
    raw = [r.raw_score for r in rows]
    hi = max(raw)
    lo = min(raw)
    if hi == lo:
        return [0.9000 for _ in rows]
    scores = []
    for value in raw:
        scaled = 0.62 + 0.369 * ((value - lo) / (hi - lo))
        scores.append(round(scaled, 4))
    for idx in range(1, len(scores)):
        if scores[idx] > scores[idx - 1]:
            scores[idx] = scores[idx - 1]
    return scores


def write_csv(rows: list[RankedCandidate], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scores = normalized_scores(rows)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (row, score) in enumerate(zip(rows, scores), start=1):
            writer.writerow([row.candidate_id, rank, f"{score:.4f}", row.reasoning])


def col_name(index: int) -> str:
    name = ""
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def sheet_xml(rows: list[list[Any]]) -> str:
    out = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>',
        '<cols><col min="1" max="1" width="16" customWidth="1"/><col min="2" max="3" width="10" customWidth="1"/><col min="4" max="4" width="110" customWidth="1"/></cols>',
        "<sheetData>",
    ]
    for r_idx, row in enumerate(rows, start=1):
        out.append(f'<row r="{r_idx}">')
        for c_idx, value in enumerate(row, start=1):
            ref = f"{col_name(c_idx)}{r_idx}"
            if isinstance(value, (int, float)) and c_idx != 1:
                out.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                escaped = html.escape(str(value), quote=False)
                out.append(f'<c r="{ref}" t="inlineStr"><is><t>{escaped}</t></is></c>')
        out.append("</row>")
    out.extend(["</sheetData>", "</worksheet>"])
    return "".join(out)


def write_xlsx_from_csv(csv_path: Path, xlsx_path: Path) -> None:
    rows: list[list[Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            converted: list[Any] = []
            for idx, value in enumerate(row):
                if row and row[0] == "candidate_id":
                    converted.append(value)
                elif idx == 1:
                    converted.append(int(value))
                elif idx == 2:
                    converted.append(float(value))
                else:
                    converted.append(value)
            rows.append(converted)

    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(xlsx_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>""")
        zf.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""")
        zf.writestr("xl/workbook.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="ranked_candidates" sheetId="1" r:id="rId1"/></sheets>
</workbook>""")
        zf.writestr("xl/_rels/workbook.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""")
        zf.writestr("xl/worksheets/sheet1.xml", sheet_xml(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank Redrob Track 1 candidates.")
    parser.add_argument("--candidates", default="data/candidates.jsonl", help="Path to candidates.jsonl")
    parser.add_argument("--out", default="outputs/submission.csv", help="CSV output path")
    parser.add_argument("--xlsx", default="outputs/ranked_candidates.xlsx", help="Optional XLSX output path")
    parser.add_argument("--limit", type=int, default=100, help="Number of candidates to output")
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        raise SystemExit(f"Candidates file not found: {candidates_path}")

    rows = select_top(candidates_path, args.limit)
    write_csv(rows, Path(args.out))
    if args.xlsx:
        write_xlsx_from_csv(Path(args.out), Path(args.xlsx))

    print(f"Wrote {len(rows)} ranked candidates to {args.out}")
    if args.xlsx:
        print(f"Wrote XLSX copy to {args.xlsx}")


if __name__ == "__main__":
    main()
