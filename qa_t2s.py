#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from t2s_ask import answer_question


ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "output" / "qa"
REPORT_MD = REPORT_DIR / "qa_report.md"
REPORT_JSON = REPORT_DIR / "qa_report.json"

QUESTIONS = [
    {
        "id": "domain-dvp",
        "question": "Explica el ciclo de liquidacion DvP en T2S",
        "must_contain_any": ["settlement", "liquid", "dvp", "delivery"],
        "min_citations": 1,
    },
    {
        "id": "domain-sdd",
        "question": "Que son los Scope Defining Documents de T2S?",
        "must_contain_any": ["scope", "sdd", "document"],
        "min_citations": 1,
    },
    {
        "id": "domain-uhb",
        "question": "Que cubre la UHB de T2S?",
        "must_contain_any": ["uhb", "user handbook", "gui"],
        "min_citations": 1,
    },
    {
        "id": "domain-connectivity",
        "question": "Como funciona la conectividad ESMIG en T2S?",
        "must_contain_any": ["esmig", "connect", "conect"],
        "min_citations": 1,
    },
    {
        "id": "domain-cr",
        "question": "Que change requests hay para T2S?",
        "must_contain_any": ["change request", "cr", "release"],
        "min_citations": 1,
    },
    {
        "id": "domain-message",
        "question": "Que documentacion local hay sobre mensajes sese en T2S?",
        "must_contain_any": ["sese", "message", "schema", "iso"],
        "min_citations": 1,
    },
]


def run_case(case: dict) -> dict:
    payload = answer_question(case["question"], top_k=10, language="es", generate=False, model_preset="local_rag")
    text = str(payload.get("answer") or "").lower()
    citations = payload.get("citations") or []
    contains = any(term.lower() in text for term in case.get("must_contain_any", []))
    enough_citations = len(citations) >= int(case.get("min_citations", 0))
    ok = contains and enough_citations and payload.get("confidence") != "low"
    return {
        "id": case["id"],
        "question": case["question"],
        "ok": ok,
        "confidence": payload.get("confidence"),
        "citations": len(citations),
        "contains_expected_term": contains,
        "answer_excerpt": str(payload.get("answer") or "")[:800],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run smoke QA checks against the local T2S GPT RAG.")
    parser.parse_args(argv)
    results = [run_case(case) for case in QUESTIONS]
    passed = sum(1 for item in results if item["ok"])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps({"passed": passed, "total": len(results), "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# T2S GPT QA Report",
        "",
        f"- Passed: {passed}/{len(results)}",
        "",
    ]
    for item in results:
        mark = "PASS" if item["ok"] else "FAIL"
        lines.append(f"## {mark} {item['id']}")
        lines.append("")
        lines.append(f"- Question: {item['question']}")
        lines.append(f"- Confidence: {item['confidence']}")
        lines.append(f"- Citations: {item['citations']}")
        lines.append("")
        lines.append(item["answer_excerpt"].replace("\n", " "))
        lines.append("")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"{passed}/{len(results)} passed")
    print(f"report: {REPORT_MD}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
