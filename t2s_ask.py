#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import linear_kernel


ROOT = Path(__file__).resolve().parent
INDEX_PATH = ROOT / "data" / "processed" / "index.pkl"
CODEX = shutil.which("codex.cmd") or shutil.which("codex") or "codex"
MODEL_PRESETS = {
    "codex_high": {"label": "Codex High", "reasoning": "high"},
    "codex_fast": {"label": "Codex Fast", "reasoning": "low"},
    "local_rag": {"label": "Solo RAG", "reasoning": "none"},
}
GENERATION_CONTEXT_HITS = 24
GENERATION_RETRIEVAL_HITS = 40
MAX_CONTEXT_HITS = 64
RELEASE_RE = re.compile(r"R(20\d{2})[._ -]?(NOV|OCT|JUN|MAR)", re.I)
MESSAGE_RE = re.compile(r"\b(acmt|admi|camt|pacs|reda|semt|sese|seev)\.(\d{3})(?:\.(\d{3}))?\b", re.I)
ACRONYM_RE = re.compile(r"\b[A-Z0-9]{2,12}\b")

SPANISH_LANGUAGE_HINTS = {
    "como",
    "cual",
    "cuales",
    "cuál",
    "cuáles",
    "dame",
    "de",
    "del",
    "el",
    "en",
    "es",
    "explica",
    "hay",
    "la",
    "las",
    "los",
    "que",
    "qué",
    "son",
    "una",
}
ENGLISH_LANGUAGE_HINTS = {"are", "does", "explain", "how", "in", "is", "of", "the", "what", "which", "who"}
QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "como",
    "con",
    "cual",
    "cuales",
    "cuál",
    "cuáles",
    "de",
    "del",
    "dame",
    "el",
    "en",
    "es",
    "explica",
    "for",
    "is",
    "la",
    "las",
    "los",
    "para",
    "que",
    "qué",
    "sobre",
    "the",
    "what",
    "which",
    "y",
}
FAMILY_HINTS = {
    "t2s_sdd": ["sdd", "scope defining", "scope", "legal basis", "service description"],
    "t2s_udfs": ["udfs", "functional specification", "message", "schema", "xsd", "xml"],
    "t2s_uhb": ["uhb", "user handbook", "gui", "screen", "u2a", "user interface"],
    "change_requests": ["change request", "cr", "crs", "release", "impact", "status"],
    "connectivity": ["connectivity", "esmig", "nsp", "swift", "message exchange", "technical"],
    "business_processes": ["business process", "bpd", "settlement", "matching", "lifecycle"],
    "migration_and_testing": ["migration", "testing", "readiness", "test", "certification"],
    "pricing": ["pricing", "fee", "tariff", "billing", "cost"],
    "legal": ["legal", "framework agreement", "guideline", "terms"],
    "participation": ["csd", "dcp", "participant", "onboarding", "account", "party"],
    "shared_features": ["shared", "common", "crdm", "billing", "reference data"],
    "messages_and_schemas": ["iso 20022", "message", "schema", "xsd", "semt", "sese", "camt", "reda"],
}


@dataclass
class Hit:
    rank: int
    score: float
    chunk: dict[str, Any]
    reason: str = ""

    @property
    def citation(self) -> str:
        return cite_label(self)


def detect_question_language(query: str) -> str:
    lower = query.lower()
    if re.search(r"[¿¡áéíóúñ]", lower):
        return "es"
    tokens = set(re.findall(r"\b[\wáéíóúñ]+\b", lower, flags=re.I))
    spanish_score = len(tokens & SPANISH_LANGUAGE_HINTS)
    english_score = len(tokens & ENGLISH_LANGUAGE_HINTS)
    return "en" if english_score > spanish_score else "es"


@lru_cache(maxsize=2)
def load_index(path: Path = INDEX_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Index not found at {path}. Run `python t2s_ingest.py` first.")
    with path.open("rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=8)
def load_json(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_query(query: str) -> str:
    query = query.strip()
    replacements = {
        "liquidacion": "settlement securities settlement delivery versus payment dvp",
        "liquidación": "settlement securities settlement delivery versus payment dvp",
        "valores": "securities settlement t2s financial instruments",
        "conectividad": "connectivity ESMIG network service provider NSP",
        "pantallas": "UHB GUI screen user interface",
        "manual": "user handbook UHB",
        "requisitos": "requirements specifications SDD UDFS",
        "mensajes": "messages schemas ISO 20022",
        "mensaje": "message schema ISO 20022",
        "esquema": "schema xsd usage guideline",
        "esquemas": "schemas xsd usage guidelines",
        "campo": "field element path minOccurs maxOccurs",
        "campos": "fields elements paths minOccurs maxOccurs",
        "cr": "change request release impact status",
        "crs": "change requests release impact status",
        "dvp": "delivery versus payment securities settlement",
        "dca": "dedicated cash account cash settlement",
        "csd": "central securities depository participant",
        "dcp": "directly connected party participant connectivity",
        "autocolateralizacion": "auto-collateralisation auto collateralisation",
        "autocolateralización": "auto-collateralisation auto collateralisation",
    }
    tokens = re.findall(r"[\w./-]+", query, flags=re.UNICODE)
    cleaned = " ".join(token for token in tokens if token.lower() not in QUESTION_STOPWORDS)
    expanded = [cleaned or query, "T2S TARGET2-Securities securities settlement CSD DCP UDFS UHB SDD ISO 20022"]
    low = query.lower()
    for key, value in replacements.items():
        if key in low:
            expanded.append(value)
    for code in query_message_codes(query):
        expanded.append(code)
        expanded.append(code.replace(".", "_"))
    release = query_release(query)
    if release:
        expanded.append(release)
    return " ".join(expanded)


def query_release(query: str) -> str:
    match = RELEASE_RE.search(query)
    if match:
        return f"R{match.group(1)}.{match.group(2).upper()}"
    reverse = re.search(r"\b(NOV|OCT|JUN|MAR)[._ -]?(20\d{2})\b", query, re.I)
    if reverse:
        return f"R{reverse.group(2)}.{reverse.group(1).upper()}"
    return ""


def query_message_codes(query: str) -> list[str]:
    codes: list[str] = []
    for match in MESSAGE_RE.finditer(query):
        code = f"{match.group(1).lower()}.{match.group(2)}"
        if match.group(3):
            code = f"{code}.{match.group(3)}"
        if code not in codes:
            codes.append(code)
    return codes


def query_acronyms(query: str) -> list[str]:
    return [item for item in ACRONYM_RE.findall(query.upper()) if item not in {"THE", "AND", "FOR"}]


def is_domain_query(query: str) -> bool:
    low = query.lower()
    if "t2s" in low or "target2-securities" in low or "target2 securities" in low:
        return True
    if query_release(query) or query_message_codes(query):
        return True
    domain_terms = [
        "securities settlement",
        "settlement",
        "dvp",
        "csd",
        "dcp",
        "dca",
        "uhb",
        "udfs",
        "sdd",
        "esmig",
        "auto-collateralisation",
        "autocolateral",
        "matching",
        "partial settlement",
        "corporate action",
        "liquidacion",
        "liquidación",
        "valores",
    ]
    return any(term in low for term in domain_terms) or any(acr in {"T2S", "CSD", "DCP", "DCA", "UDFS", "UHB", "SDD"} for acr in query_acronyms(query))


def _bm25_scores(index: dict[str, Any], query: str) -> np.ndarray:
    vectorizer = index.get("bm25_vectorizer")
    matrix = index.get("bm25_matrix")
    idf = index.get("bm25_idf")
    doc_len = index.get("bm25_doc_len")
    avgdl = float(index.get("bm25_avgdl") or 1.0)
    if vectorizer is None or matrix is None or idf is None or doc_len is None:
        return np.zeros(len(index.get("chunks", [])), dtype=float)
    q = vectorizer.transform([query])
    if q.nnz == 0:
        return np.zeros(matrix.shape[0], dtype=float)
    k1 = 1.5
    b = 0.75
    scores = np.zeros(matrix.shape[0], dtype=float)
    for term_idx in q.indices:
        col = matrix[:, term_idx].tocoo()
        if col.nnz == 0:
            continue
        freq = col.data.astype(float)
        denom = freq + k1 * (1 - b + b * (doc_len[col.row] / avgdl))
        scores[col.row] += idf[term_idx] * (freq * (k1 + 1) / denom)
    max_score = float(scores.max() or 0.0)
    if max_score:
        scores = scores / max_score
    return scores


def _chunk_text_for_ranking(chunk: dict[str, Any]) -> str:
    return "\n".join(
        str(chunk.get(key) or "")
        for key in [
            "title",
            "family",
            "category",
            "release",
            "unit_type",
            "unit",
            "message_id",
            "usage_guideline_name",
            "collection",
            "local_path",
            "source_url",
            "text",
        ]
    )


def metadata_bonus(chunk: dict[str, Any], query: str) -> float:
    low = query.lower()
    hay = _chunk_text_for_ranking(chunk).lower()
    bonus = 0.0
    family = str(chunk.get("family") or "")
    for hinted_family, hints in FAMILY_HINTS.items():
        if family == hinted_family and any(hint in low for hint in hints):
            bonus += 0.18
    release = query_release(query)
    if release:
        bonus += 0.35 if chunk.get("release") == release else -0.05
    for code in query_message_codes(query):
        aliases = {code, code.replace(".", "_"), code.replace(".", " ")}
        if any(alias in hay for alias in aliases):
            bonus += 0.35
    for acronym in query_acronyms(query):
        if re.search(rf"\b{re.escape(acronym.lower())}\b", hay):
            bonus += 0.08
    if chunk.get("revision_status") == "clean":
        bonus += 0.04
    return bonus


def retrieve(index: dict[str, Any], query: str, top_k: int = 8, pool: int = 180) -> list[Hit]:
    chunks = index.get("chunks") or []
    if not chunks:
        return []
    normalized = normalize_query(query)
    word_scores = linear_kernel(index["word_vectorizer"].transform([normalized]), index["word_matrix"]).ravel()
    char_scores = linear_kernel(index["char_vectorizer"].transform([normalized]), index["char_matrix"]).ravel()
    bm25_scores = _bm25_scores(index, normalized)
    scores = 0.48 * word_scores + 0.25 * char_scores + 0.27 * bm25_scores
    pool_size = min(max(pool, top_k * 6), len(chunks))
    candidate_idx = np.argpartition(scores, -pool_size)[-pool_size:]
    ranked = sorted(candidate_idx, key=lambda idx: scores[idx] + metadata_bonus(chunks[idx], query), reverse=True)
    hits: list[Hit] = []
    for rank, idx in enumerate(ranked[:top_k], start=1):
        score = float(scores[idx] + metadata_bonus(chunks[idx], query))
        hits.append(Hit(rank=rank, score=score, chunk=chunks[int(idx)], reason="hybrid"))
    return hits


def augment_hits(index: dict[str, Any], query: str, hits: list[Hit]) -> list[Hit]:
    return hits


def expand_neighbor_hits(index: dict[str, Any], hits: list[Hit], max_neighbors: int = 1, max_total: int = MAX_CONTEXT_HITS) -> list[Hit]:
    chunks = index.get("chunks") or []
    chunk_id_to_pos = index.get("chunk_id_to_pos") or {}
    selected: list[Hit] = []
    seen: set[str] = set()

    def add(hit: Hit) -> None:
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit.chunk))
        if chunk_id not in seen and len(selected) < max_total:
            selected.append(hit)
            seen.add(chunk_id)

    for hit in hits:
        add(hit)
        pos = chunk_id_to_pos.get(hit.chunk.get("chunk_id"))
        if pos is None:
            continue
        for offset in range(1, max_neighbors + 1):
            for neighbor_pos in (pos - offset, pos + offset):
                if 0 <= neighbor_pos < len(chunks):
                    neighbor = chunks[neighbor_pos]
                    if neighbor.get("doc_id") == hit.chunk.get("doc_id"):
                        add(Hit(rank=len(selected) + 1, score=max(hit.score - 0.08 * offset, 0.01), chunk=neighbor, reason="neighbor"))
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
    return selected


def context_pointer_priority(query: str, hit: Hit) -> tuple[float, float]:
    chunk = hit.chunk
    hay = _chunk_text_for_ranking(chunk).lower()
    priority = metadata_bonus(chunk, query)
    if hit.reason == "neighbor":
        priority -= 0.08
    query_terms = [term for term in re.findall(r"[\w./-]{3,}", query.lower()) if term not in QUESTION_STOPWORDS]
    priority += min(sum(0.025 for term in query_terms if term in hay), 0.5)
    return priority, hit.score


def rerank_context_hits(query: str, hits: list[Hit], max_total: int | None = None) -> list[Hit]:
    ranked = sorted(enumerate(hits), key=lambda item: (*context_pointer_priority(query, item[1]), -item[0]), reverse=True)
    selected: list[Hit] = []
    seen_chunks: set[str] = set()
    seen_texts: set[str] = set()
    for _, hit in ranked:
        chunk_id = str(hit.chunk.get("chunk_id") or id(hit.chunk))
        text_key = re.sub(r"\W+", " ", str(hit.chunk.get("text") or "").lower())[:700]
        if chunk_id in seen_chunks or (text_key and text_key in seen_texts):
            continue
        seen_chunks.add(chunk_id)
        if text_key:
            seen_texts.add(text_key)
        selected.append(hit)
        if max_total and len(selected) >= max_total:
            break
    for rank, hit in enumerate(selected, start=1):
        hit.rank = rank
    return selected


def trim_excerpt(text: str, max_chars: int = 950) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    last_stop = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(": "))
    if last_stop > 350:
        return cut[: last_stop + 1].strip()
    return cut.rstrip() + "..."


def cite_label(hit: Hit) -> str:
    chunk = hit.chunk
    title = chunk.get("title") or "Untitled"
    release = chunk.get("release")
    unit_type = chunk.get("unit_type") or "unit"
    unit = chunk.get("unit")
    release_part = f", {release}" if release else ""
    where = f"{unit_type} {unit}" if unit not in (None, "") else unit_type
    return f"{title}{release_part}, {where}"


def citations_from_hits(hits: list[Hit]) -> list[dict[str, Any]]:
    citations = []
    for n, hit in enumerate(hits, start=1):
        citations.append(
            {
                "n": n,
                "title": hit.chunk.get("title"),
                "release": hit.chunk.get("release"),
                "family": hit.chunk.get("family"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "score": round(hit.score, 4),
                "label": cite_label(hit),
            }
        )
    return citations


def build_answer(query: str, hits: list[Hit], language: str = "es") -> dict[str, Any]:
    if not is_domain_query(query):
        msg = (
            "I cannot see a T2S term, message, acronym, release or process in that question. Ask with the specific T2S concept and I will answer from the local corpus."
            if language == "en"
            else "No veo un termino, mensaje, acronimo, release o proceso T2S en esa pregunta. Pon el concepto T2S concreto y respondo con el corpus local."
        )
        return {"answer": msg, "citations": [], "confidence": "low", "skip_generation": True}
    if not hits:
        msg = "I cannot find enough evidence in the local T2S index." if language == "en" else "No aparece evidencia suficiente en el indice local de T2S."
        return {"answer": msg, "citations": [], "confidence": "low"}
    top = hits[: min(6, len(hits))]
    intro = (
        "These are the local passages I would inspect first:"
        if language == "en"
        else "Estos son los pasajes locales que revisaria primero:"
    )
    bullets = [f"{n}. {trim_excerpt(hit.chunk.get('text', ''))} [{n}]" for n, hit in enumerate(top, start=1)]
    citations = citations_from_hits(top)
    source_title = "References" if language == "en" else "Referencias"
    answer = intro + "\n\n" + "\n\n".join(bullets)
    answer += f"\n\n{source_title}:\n" + "\n".join(f"[{c['n']}] {c['label']} - {c['local_path']}" for c in citations)
    confidence = "high" if hits[0].score >= 0.25 else "medium" if hits[0].score >= 0.12 else "low"
    return {"answer": answer, "citations": citations, "confidence": confidence}


def prioritize_generation_hits(query: str, hits: list[Hit], max_hits: int) -> list[Hit]:
    reranked = rerank_context_hits(query, hits, max_total=max(len(hits), max_hits))
    return reranked[:max_hits]


def build_codex_context(query: str, hits: list[Hit], max_hits: int = GENERATION_CONTEXT_HITS) -> dict[str, Any]:
    hits = prioritize_generation_hits(query, hits, max_hits=max_hits)
    evidence = []
    for n, hit in enumerate(hits[:max_hits], start=1):
        evidence.append(
            {
                "ref": n,
                "score": round(hit.score, 4),
                "retrieval_reason": hit.reason or "hybrid",
                "citation": cite_label(hit),
                "title": hit.chunk.get("title"),
                "family": hit.chunk.get("family"),
                "release": hit.chunk.get("release"),
                "unit_type": hit.chunk.get("unit_type"),
                "unit": hit.chunk.get("unit"),
                "local_path": hit.chunk.get("local_path"),
                "source_url": hit.chunk.get("source_url"),
                "excerpt": trim_excerpt(hit.chunk.get("text", ""), max_chars=3000),
            }
        )
    return {
        "question": query,
        "retrieval_pipeline": "hybrid TF-IDF word + TF-IDF char + BM25 + metadata boosts + local rerank + neighbor expansion",
        "instructions_for_codex_high": (
            "Answer using only the local T2S evidence dossier. Cite every substantive claim with [n]. "
            "If evidence is weak or absent, say so."
        ),
        "evidence": evidence,
    }


def format_chat_history(chat_history: list[dict[str, str]] | None, language: str) -> str:
    if not chat_history:
        return ""
    labels = {"user": "Usuario" if language == "es" else "User", "assistant": "Asistente" if language == "es" else "Assistant"}
    lines: list[str] = []
    for turn in chat_history[-10:]:
        role = str(turn.get("role", "")).lower()
        content = re.sub(r"\s+", " ", str(turn.get("content", ""))).strip()
        if role in labels and content:
            lines.append(f"{labels[role]}: {content[:1600]}")
    if not lines:
        return ""
    title = "Conversacion reciente:" if language == "es" else "Recent conversation:"
    return title + "\n" + "\n".join(lines)


def build_generation_prompt(
    query: str,
    hits: list[Hit],
    language: str = "es",
    max_hits: int = GENERATION_CONTEXT_HITS,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    if language == "auto":
        language = detect_question_language(query)
    lang_name = "Spanish" if language == "es" else "English"
    context = build_codex_context(query, hits, max_hits=max_hits)
    evidence_lines = []
    for item in context["evidence"]:
        evidence_lines.append(
            "\n".join(
                [
                    f"[{item['ref']}] {item['citation']}",
                    f"Retrieval: {item['retrieval_reason']} | score={item['score']}",
                    f"Local path: {item['local_path']}",
                    f"Source URL: {item['source_url']}",
                    f"Excerpt: {item['excerpt']}",
                ]
            )
        )
    history_block = format_chat_history(chat_history, language)
    style_rules = (
        "Write in Spanish with a natural technical-assistant voice. Start with the answer, then develop it. "
        "Use official T2S names in English when the documents use them. If something is missing, say that it does not appear in the retrieved local documentation."
        if language == "es"
        else "Write in English with a natural, direct technical-assistant voice. Start with the answer, then develop it. "
        "Use official T2S names. If something is missing, say that it does not appear in the retrieved local documentation."
    )
    return f"""You are a senior T2S documentation assistant running inside a local T2S documentation repository.

Answer the user's question in {lang_name}.
Use only the local T2S corpus: the evidence dossier below and the listed local paths as read-only pointers. Do not invent facts and do not use the internet.
Every substantive claim must cite evidence with [n]. Put references at the end with title, page/unit, and local path.
Do not dump raw excerpts. Synthesize the answer.
{style_rules}

User question:
{query}

{history_block + chr(10) if history_block else ""}Local evidence dossier:

{chr(10).join(evidence_lines)}
"""


def generate_with_codex(
    query: str,
    hits: list[Hit],
    language: str = "es",
    timeout: int | None = None,
    chat_history: list[dict[str, str]] | None = None,
    model_preset: str = "codex_high",
) -> str:
    if os.environ.get("T2S_DISABLE_CODEX", "").lower() in {"1", "true", "yes"}:
        raise RuntimeError("Codex generation disabled by T2S_DISABLE_CODEX")
    if not shutil.which("codex.cmd") and not shutil.which("codex"):
        raise RuntimeError("codex CLI not found")
    prompt = build_generation_prompt(query, hits, language=language, max_hits=GENERATION_CONTEXT_HITS, chat_history=chat_history)
    timeout = timeout or int(os.environ.get("T2S_CODEX_TIMEOUT", "180"))
    preset = MODEL_PRESETS.get(model_preset, MODEL_PRESETS["codex_high"])
    reasoning = preset["reasoning"]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        output_path = Path(tmp.name)
    cmd = [
        CODEX,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "-s",
        "read-only",
        "--color",
        "never",
        "-c",
        f'model_reasoning_effort="{reasoning}"',
        "-o",
        str(output_path),
        "-",
    ]
    model = os.environ.get("T2S_CODEX_MODEL", "").strip()
    if model:
        cmd[2:2] = ["-m", model]
    try:
        result = subprocess.run(cmd, input=prompt, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=timeout, cwd=str(ROOT), check=False)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "").strip()[:1200])
        answer = output_path.read_text(encoding="utf-8", errors="replace").strip()
        if not answer:
            raise RuntimeError("codex returned an empty answer")
        return answer
    finally:
        try:
            output_path.unlink(missing_ok=True)
        except Exception:
            pass


def _retrieve_context(query: str, top_k: int, generate: bool) -> list[Hit]:
    index = load_index()
    retrieval_k = max(top_k, GENERATION_RETRIEVAL_HITS) if generate else top_k
    hits = retrieve(index, query, top_k=retrieval_k)
    hits = augment_hits(index, query, hits)
    max_context = max(retrieval_k + 16, top_k, GENERATION_CONTEXT_HITS if generate else top_k)
    hits = expand_neighbor_hits(index, hits, max_neighbors=1, max_total=min(max_context, MAX_CONTEXT_HITS))
    return rerank_context_hits(query, hits, max_total=min(max_context, MAX_CONTEXT_HITS))


def answer_question(
    query: str,
    top_k: int = 16,
    language: str = "auto",
    generate: bool = False,
    retrieval_query: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    model_preset: str = "codex_high",
) -> dict[str, Any]:
    resolved_language = detect_question_language(query) if language == "auto" else language
    search_query = retrieval_query or query
    if model_preset == "local_rag":
        generate = False
    hits = _retrieve_context(search_query, top_k=top_k, generate=generate)
    payload = build_answer(query, hits, language=resolved_language)
    if generate and hits and payload.get("confidence") != "low":
        generation_hits = prioritize_generation_hits(search_query, hits, max_hits=GENERATION_CONTEXT_HITS)
        try:
            payload["answer"] = generate_with_codex(query, generation_hits, language=resolved_language, chat_history=chat_history, model_preset=model_preset)
            payload["citations"] = citations_from_hits(generation_hits[:GENERATION_CONTEXT_HITS])
            payload["generated_by"] = MODEL_PRESETS.get(model_preset, MODEL_PRESETS["codex_high"])["label"].lower().replace(" ", "_")
        except Exception as exc:
            payload["generated_by"] = "fallback_extractivo"
            payload["generator_error"] = str(exc)
    elif payload.get("skip_generation"):
        payload["generated_by"] = "structured"
    elif model_preset == "local_rag":
        payload["generated_by"] = "local_rag"
    payload["question"] = query
    payload["language"] = resolved_language
    payload["model"] = model_preset
    payload["hits"] = [
        {
            "rank": hit.rank,
            "score": round(hit.score, 4),
            "reason": hit.reason,
            "citation": cite_label(hit),
            "chunk": {
                key: hit.chunk.get(key)
                for key in [
                    "chunk_id",
                    "doc_id",
                    "title",
                    "category",
                    "family",
                    "release",
                    "revision_status",
                    "unit_type",
                    "unit",
                    "local_path",
                    "source_url",
                    "context_path",
                ]
            },
            "excerpt": trim_excerpt(hit.chunk.get("text", ""), max_chars=900),
        }
        for hit in hits
    ]
    return payload


def read_question(args: argparse.Namespace) -> str:
    if args.question:
        return " ".join(args.question).strip()
    return sys.stdin.read().strip()


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Ask questions against the local T2S index.")
    parser.add_argument("question", nargs="*", help="question; if omitted, stdin is used")
    parser.add_argument("--json", action="store_true", help="print full JSON answer")
    parser.add_argument("--context", action="store_true", help="print optimized evidence JSON for Codex High")
    parser.add_argument("--generate", action="store_true", help="generate a conversational answer with Codex High")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lang", choices=["auto", "es", "en"], default="auto")
    parser.add_argument("--model", choices=sorted(MODEL_PRESETS), default="codex_high", help="generation preset")
    args = parser.parse_args(argv)
    query = read_question(args)
    if not query:
        print("ERROR: empty question", file=sys.stderr)
        return 2
    try:
        hits = _retrieve_context(query, top_k=args.top_k, generate=args.generate)
        if args.context:
            print(json.dumps(build_codex_context(query, hits, max_hits=args.top_k), ensure_ascii=False, indent=2))
            return 0
        payload = answer_question(query, top_k=args.top_k, language=args.lang, generate=args.generate, model_preset=args.model)
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print(payload["answer"])
        return 0
    except Exception as exc:
        language = detect_question_language(query)
        message = (
            f"I could not complete this query, but the CLI stayed alive. Try a more specific T2S term or run with --context. Error: {exc}"
            if language == "en"
            else f"No he podido completar esta consulta, pero el CLI sigue vivo. Prueba con un termino T2S mas concreto o ejecuta con --context. Error: {exc}"
        )
        if args.json:
            print(json.dumps({"question": query, "answer": message, "error": str(exc), "confidence": "low"}, ensure_ascii=False, indent=2))
            return 0
        print(message)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
