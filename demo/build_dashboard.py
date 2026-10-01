"""Build demo/dashboard.html (pure HTML + CSS, no JavaScript) from the saved artifacts.

Reads golden_dataset.json + artifacts/*.json, reuses the evaluation core in
template.py (find_root_cause, rerank_by_overlap, retrieval metrics), runs the
unit tests and the dataset validator, renders static HTML and drops it into
dashboard.template.html. No API calls.

Run from the repo root:  python demo/build_dashboard.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from template import (  # noqa: E402
    EvalResult,
    FailureAnalyzer,
    QAPair,
    RAGASEvaluator,
    rerank_by_overlap,
)

HERE = Path(__file__).resolve().parent
DIFFS = ["easy", "medium", "hard", "adversarial"]
DIFF_LABEL = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "adversarial": "Adversarial"}

# 5 Whys summaries from reflection.md for the three lowest cases.
DIAGNOSIS = {
    "A01": (
        "Không từ chối đúng scope: không giải thích vai trò, không gợi ý chủ đề OrbitTech. Cả 3 answer metrics bằng 0.",
        "Quy tắc scope chỉ nằm trong corpus. BM25 không khớp “invest/stocks” với “investment advice” nên không lấy được 00_system_scope.md.",
        "Đưa quy tắc scope và prompt-injection vào system prompt cố định; thêm bước phân loại scope trước retrieval.",
        "Đồng ý một phần: lỗi bắt đầu ở retrieval và bị prompt khuếch đại. Answer không chứa claim bịa.",
    ),
    "H01": (
        "Kết luận đúng (7 ngày, phí 15%) nhưng Completeness 0.243 vì thiếu lý do áp dụng Return Policy v1.0.",
        "Prompt yêu cầu “Answer concisely” và không yêu cầu nêu policy version khi câu hỏi có ngày tháng.",
        "Prompt: nêu version → lý do → kết quả cho câu hỏi phụ thuộc ngày. Thêm LLM judge theo rubric 3.3.",
        "Đồng ý vế “improve generation”, không đồng ý “increase context window” vì Recall 0.946, Precision 1.0.",
    ),
    "E04": (
        "Câu Easy bị fail: answer đủ ý (Completeness 0.880) nhưng thêm 2 quyền lợi lấy từ chunk khác, Faithfulness 0.359.",
        "Model gom mọi chunk liên quan; Faithfulness đo so với gold context nên phạt cả thông tin đúng nằm ngoài gold.",
        "Prompt: chỉ trả lời điều được hỏi. Evaluation: đo faithfulness so với retrieved contexts.",
        "Không đồng ý “improve retrieval”: Recall 0.960 và Precision 1.0 cho thấy retrieval tốt.",
    ),
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(args: list[str]) -> str:
    done = subprocess.run(
        [sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return (done.stdout + done.stderr).strip()


def f3(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def collect() -> dict:
    golden = {p["id"]: p for p in _load(ROOT / "golden_dataset.json")["qa_pairs"]}
    actual = _load(ROOT / "artifacts" / "actual_answers.json")
    bench = _load(ROOT / "artifacts" / "benchmark_results.json")
    answers = {a["id"]: a for a in actual["answers"]}

    evaluator, analyzer = RAGASEvaluator(), FailureAnalyzer()
    cases, rerank = [], []
    for r in bench["results"]:
        gold, ans = golden[r["id"]], answers[r["id"]]
        expected = gold["expected_answer"]
        result = EvalResult(
            QAPair(r["question"], expected), r["actual_answer"],
            r["faithfulness"], r["relevance"], r["completeness"], r["passed"], r["failure_type"],
        )
        chunks = [c["text"] for c in ans["retrieved_contexts"]]
        reranked = rerank_by_overlap(chunks, r["question"])
        rerank.append({
            "id": r["id"],
            "recall_before": evaluator.evaluate_context_recall(chunks, expected),
            "recall_after": evaluator.evaluate_context_recall(reranked, expected),
            "p_before": evaluator.evaluate_context_precision(chunks, expected),
            "p_after": evaluator.evaluate_context_precision(reranked, expected),
        })
        cases.append({
            **r,
            "attack_type": gold["attack_type"],
            "expected_answer": expected,
            "gold": gold["contexts"],
            "retrieved": ans["retrieved_contexts"],
            "root_cause": None if r["passed"] else analyzer.find_root_cause(result),
        })

    tests = _run(["-m", "pytest", "tests/", "-q"]).splitlines()
    return {
        "agent": actual["agent"],
        "generated_at": actual["generated_at"],
        "summary": bench["summary"],
        "cases": cases,
        "rerank": rerank,
        "tests": tests[-1].strip("= ") if tests else "n/a",
        "validator_pass": "PASS:" in _run(["validate_golden_dataset.py"]),
        "doc_coverage": len({c["source_doc"] for p in golden.values() for c in p["contexts"]}),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def meter(value: float | None, label: str) -> str:
    """Compact score meter used in each case row."""
    if value is None:
        return f'<span class="m"><span class="m-l">{label}</span><span class="m-v">—</span></span>'
    low = " low" if value < 0.5 else ""
    return (
        f'<span class="m{low}" title="{label}: {value:.3f}"><span class="m-l">{label}</span>'
        f'<span class="m-v">{value:.3f}</span><span class="m-b"><i style="width:{value * 100:.1f}%"></i></span></span>'
    )


def ruler_row(label: str, value: float) -> str:
    """One metric on a 0–1 measuring ruler with quality zones."""
    return f"""
      <div class="rrow">
        <span class="r-label">{label}</span>
        <div class="ruler" title="{label}: {value:.3f}">
          <span class="fill" style="width:{value * 100:.1f}%"></span>
          <span class="mark" style="left:{value * 100:.1f}%"></span>
        </div>
        <span class="r-val">{value:.3f}</span>
      </div>"""


def render_metrics(s: dict, cases: list[dict]) -> str:
    overall = sum(c["overall"] for c in cases) / len(cases)
    ticks = "".join(f'<span style="left:{t * 10}%">{t / 10:.1f}</span>' for t in range(0, 11, 2))
    return f"""
      <div class="rgroup">Retrieval</div>
      {ruler_row("Context Recall", s["avg_context_recall"])}
      {ruler_row("Context Precision", s["avg_context_precision"])}
      <div class="rgroup">Answer</div>
      {ruler_row("Faithfulness", s["avg_faithfulness"])}
      {ruler_row("Relevance", s["avg_relevance"])}
      {ruler_row("Completeness", s["avg_completeness"])}
      {ruler_row("Overall", overall)}
      <div class="rrow axis"><span></span><div class="r-ticks">{ticks}</div><span></span></div>"""


def render_units(cases: list[dict]) -> str:
    rows = []
    for d in DIFFS:
        group = [c for c in cases if c["difficulty"] == d]
        passed = sum(c["passed"] for c in group)
        cells = "".join(
            f'<a class="unit {"ok" if c["passed"] else "no"}" href="#case-{c["id"]}" '
            f'title="{c["id"]} · {"pass" if c["passed"] else c["failure_type"]} · overall {c["overall"]:.3f}">'
            f'{c["id"]}</a>'
            for c in group
        )
        rows.append(
            f'<div class="urow"><span class="u-label">{DIFF_LABEL[d]}</span>'
            f'<div class="units">{cells}</div><span class="u-val">{passed}/{len(group)}</span></div>'
        )
    return "".join(rows)


def render_failures(s: dict, cases: list[dict]) -> str:
    n_fail = s["total"] - s["passed"]
    rows = []
    for ftype, n in sorted(s["failure_types"].items(), key=lambda kv: -kv[1]):
        ids = ", ".join(c["id"] for c in cases if c["failure_type"] == ftype)
        rows.append(
            f'<div class="frow" title="{ftype}: {ids}"><span class="f-label">{ftype}</span>'
            f'<div class="f-track"><i style="width:{n / n_fail * 100:.1f}%"></i></div>'
            f'<span class="f-val">{n}</span><span class="f-ids">{ids}</span></div>'
        )
    return "".join(rows)


def render_case(c: dict, worst_id: str) -> str:
    gold_docs = {g["source_doc"] for g in c["gold"]}
    status = (
        '<span class="pill ok">✓ Pass</span>' if c["passed"]
        else f'<span class="pill no">✕ {escape(c["failure_type"] or "")}</span>'
    )
    tag = DIFF_LABEL[c["difficulty"]] + (f' · {escape(c["attack_type"])}' if c["attack_type"] else "")
    chunks = "".join(
        f"""<li class="chunk{' gold' if r['source_doc'] in gold_docs else ''}">
              <div class="c-top"><b>#{i}</b><span>{escape(r['chunk_id'])}</span><span>{escape(r['source_doc'])}</span>
              <span>BM25 {r['score']:.2f}</span>{'<span class="gold-tag">✓ nguồn gold</span>' if r['source_doc'] in gold_docs else ''}</div>
              <p>{escape(r['text'])}</p></li>"""
        for i, r in enumerate(c["retrieved"], start=1)
    )
    gold = "".join(
        f'<li class="chunk"><div class="c-top"><span>{escape(g["source_doc"])}</span></div><p>{escape(g["text"])}</p></li>'
        for g in c["gold"]
    )
    diag = ""
    if c["root_cause"] or c["id"] in DIAGNOSIS:
        rc = f'<p class="rc"><span class="eyebrow">find_root_cause()</span>{escape(c["root_cause"])}</p>' if c["root_cause"] else ""
        dl = ""
        if c["id"] in DIAGNOSIS:
            sym, root, fix, agree = DIAGNOSIS[c["id"]]
            dl = (f"<dl><dt>Triệu chứng</dt><dd>{sym}</dd><dt>Root cause</dt><dd>{root}</dd>"
                  f"<dt>Đề xuất</dt><dd>{fix}</dd><dt>So với hàm</dt><dd>{agree}</dd></dl>")
        diag = f'<div class="diag">{rc}{dl}</div>'

    return f"""
    <details class="case" id="case-{c['id']}" data-diff="{c['difficulty']}" data-pass="{'pass' if c['passed'] else 'fail'}"{' open' if c['id'] == worst_id else ''}>
      <summary>
        <span class="c-id">{c['id']}</span>
        <span class="c-q">{escape(c['question'])}</span>
        <span class="c-meters">
          {meter(c['context_recall'], 'Rec')}{meter(c['context_precision'], 'Prec')}
          {meter(c['faithfulness'], 'Faith')}{meter(c['relevance'], 'Rel')}{meter(c['completeness'], 'Comp')}
        </span>
        <span class="c-overall"><span class="eyebrow">Overall</span><b>{c['overall']:.3f}</b></span>
        <span class="c-status">{status}</span>
      </summary>
      <div class="c-body">
        <div class="c-qa">
          <p class="chip">{tag}</p>
          <div class="qa"><span class="eyebrow">Đáp án mong đợi</span><p class="ans">{escape(c['expected_answer'])}</p></div>
          <div class="qa"><span class="eyebrow">Câu trả lời thật</span><p class="ans actual">{escape(c['actual_answer'])}</p></div>
          {diag}
        </div>
        <div class="c-trace">
          <span class="eyebrow">5 chunk đã retrieve</span>
          <ol class="chunks">{chunks}</ol>
          <details class="gold-box"><summary>Gold evidence ({len(c['gold'])})</summary><ul class="chunks">{gold}</ul></details>
        </div>
      </div>
    </details>"""


def render_rerank(rows: list[dict]) -> tuple[str, str]:
    n = len(rows)
    avg = lambda k, arr: sum(r[k] for r in arr) / len(arr)  # noqa: E731
    shown = [r for r in rows if r["p_before"] < 0.9999]
    same = sum(abs(r["recall_before"] - r["recall_after"]) < 1e-9 for r in rows)
    up = sum(r["p_after"] - r["p_before"] > 1e-9 for r in shown)
    facts = f"""
      <div class="fact"><b>{avg('p_before', rows):.3f} → {avg('p_after', rows):.3f}</b><span>Context Precision trung bình {n} case</span></div>
      <div class="fact"><b>{same}/{n}</b><span>case giữ nguyên Context Recall</span></div>
      <div class="fact"><b>{up}/{len(shown)}</b><span>case có precision tăng, không case nào giảm</span></div>"""

    W, L, R, row_h, top = 640, 52, 72, 36, 8
    H = top + len(shown) * row_h + 30
    lo, hi = 0.6, 1.0
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)  # noqa: E731
    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Context Precision trước và sau rerank">']
    for t in (0.6, 0.7, 0.8, 0.9, 1.0):
        parts.append(f'<line class="g" x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top}" y2="{H - 24}"/>'
                     f'<text x="{x(t):.1f}" y="{H - 6}" text-anchor="middle">{t:.1f}</text>')
    for i, r in enumerate(shown):
        y = top + i * row_h + row_h / 2
        a, b, d = x(r["p_before"]), x(r["p_after"]), r["p_after"] - r["p_before"]
        parts.append(f'<g><title>{r["id"]}: precision {r["p_before"]:.3f} → {r["p_after"]:.3f}; recall {r["recall_before"]:.3f} (không đổi)</title>')
        parts.append(f'<text class="lbl" x="0" y="{y + 4:.1f}">{r["id"]}</text>')
        if b - a > 1:
            parts.append(f'<line class="link" x1="{a:.1f}" x2="{b:.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<circle class="before" cx="{a:.1f}" cy="{y:.1f}" r="5"/><circle class="after" cx="{b:.1f}" cy="{y:.1f}" r="6"/>')
        parts.append(f'<text class="delta" x="{W - R + 14}" y="{y + 4:.1f}">{"+" if d >= 0 else ""}{d:.3f}</text></g>')
    parts.append("</svg>")
    return facts, "".join(parts)


def render(data: dict) -> dict[str, str]:
    s, cases, agent = data["summary"], data["cases"], data["agent"]
    worst = min(cases, key=lambda c: c["overall"])
    weakest = min(
        (("Faithfulness", s["avg_faithfulness"]), ("Relevance", s["avg_relevance"]),
         ("Completeness", s["avg_completeness"])), key=lambda kv: kv[1],
    )
    ran_at = datetime.fromisoformat(data["generated_at"]).astimezone(timezone(timedelta(hours=7)))
    tests_ok = "passed" in data["tests"] and "failed" not in data["tests"]
    n_tests = data["tests"].split(" passed")[0].split()[-1] if tests_ok else "—"
    facts, dumbbell = render_rerank(data["rerank"])

    filters_diff = "".join(
        f'<input type="radio" name="fd" id="fd-{v}"{" checked" if v == "all" else ""}><label for="fd-{v}">{lab}</label>'
        for v, lab in [("all", "Tất cả"), *[(d, DIFF_LABEL[d]) for d in DIFFS]]
    )
    filters_pass = "".join(
        f'<input type="radio" name="fp" id="fp-{v}"{" checked" if v == "all" else ""}><label for="fp-{v}">{lab}</label>'
        for v, lab in [("all", "Mọi kết quả"), ("pass", "Pass"), ("fail", "Fail")]
    )

    return {
        "MODEL": escape(agent["model"]),
        "TOPK": str(agent["top_k"]),
        "PROMPT": escape(agent["prompt_version"]),
        "RAN_AT": ran_at.strftime("%d/%m/%Y %H:%M") + " GMT+7",
        "PASSED": str(s["passed"]),
        "TOTAL": str(s["total"]),
        "PASS_RATE": f"{s['pass_rate'] * 100:.0f}",
        "PRECISION": f3(s["avg_context_precision"]),
        "RECALL": f3(s["avg_context_recall"]),
        "WEAK_NAME": weakest[0],
        "WEAK_VAL": f3(weakest[1]),
        "TESTS_N": n_tests,
        "TESTS_LINE": escape(data["tests"]),
        "VALIDATOR": "PASS" if data["validator_pass"] else "FAIL",
        "DOCS": str(data["doc_coverage"]),
        "WORST_ID": worst["id"],
        "WORST_VAL": f3(worst["overall"]),
        "WORST_TYPE": escape(worst["failure_type"] or "pass"),
        "METRICS": render_metrics(s, cases),
        "UNITS": render_units(cases),
        "FAILURES": render_failures(s, cases),
        "N_FAIL": str(s["total"] - s["passed"]),
        "FILTERS_DIFF": filters_diff,
        "FILTERS_PASS": filters_pass,
        "CASES": "".join(render_case(c, worst["id"]) for c in cases),
        "RERANK_FACTS": facts,
        "RERANK_SVG": dumbbell,
    }


def main() -> None:
    data = collect()
    page = (HERE / "dashboard.template.html").read_text(encoding="utf-8")
    for key, value in render(data).items():
        page = page.replace("{{" + key + "}}", value)
    out = HERE / "dashboard.html"
    out.write_text(
        '<!doctype html>\n<html lang="vi">\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        + page,
        encoding="utf-8",
    )
    print(f"Wrote {out.relative_to(ROOT)} | tests: {data['tests']} | validator PASS: {data['validator_pass']}")


if __name__ == "__main__":
    main()
