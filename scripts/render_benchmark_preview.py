#!/usr/bin/env python3

import html
import json
from datetime import datetime
from pathlib import Path

import markdown


ROOT = Path("/Users/liyc/Desktop/nativeMLLM/evaluate_framework")
NORMALIZED = ROOT / "data" / "normalized"
OUTPUT = ROOT / "benchmark_preview.html"
SHORTLIST_MD = ROOT / "visual-benchmark-shortlist.md"


FILES = [
    ("babyvision", NORMALIZED / "babyvision" / "train.jsonl"),
    ("chartqapro", NORMALIZED / "chartqapro" / "test.jsonl"),
    ("countbench", NORMALIZED / "countbench" / "test.jsonl"),
    ("hallusionbench", NORMALIZED / "hallusionbench" / "test.sample_100.jsonl"),
    ("mathvista", NORMALIZED / "mathvista" / "testmini.jsonl"),
    ("mmmu", NORMALIZED / "mmmu" / "validation.sample_100.jsonl"),
    ("omnidocbench", NORMALIZED / "omnidocbench" / "train.sample_100.jsonl"),
    ("phyx_openended", NORMALIZED / "phyx_openended" / "test_mini.jsonl"),
    ("realworldqa", NORMALIZED / "realworldqa" / "test.jsonl"),
    ("refcoco_avg", NORMALIZED / "refcoco_avg" / "all8.sample_per_split_5.jsonl"),
    ("visulogic", NORMALIZED / "visulogic" / "test.sample_100.jsonl"),
    ("vlmsareblind", NORMALIZED / "vlmsareblind" / "valid.jsonl"),
]


def load_two(path):
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        all_rows = [json.loads(line) for line in f]
    with_media = [row for row in all_rows if row.get("media")]
    if len(with_media) >= 2:
        return with_media[:2]
    if len(with_media) == 1:
        for row in all_rows:
            if row is not with_media[0]:
                return [with_media[0], row]
    return all_rows[:2]


def rel_media_src(media_path):
    target = ROOT / media_path
    return target.relative_to(ROOT).as_posix() if target.exists() else None


def render_choices(prompt):
    choices = prompt.get("choices") or []
    labels = prompt.get("choice_labels") or []
    if not choices:
        return ""
    items = []
    for i, choice in enumerate(choices):
        label = labels[i] if i < len(labels) else chr(ord("A") + i)
        items.append(f"<li><b>{html.escape(str(label))}.</b> {html.escape(str(choice))}</li>")
    return "<div class='choices'><div class='sec-title'>Choices</div><ol>" + "".join(items) + "</ol></div>"


def render_answer(answer):
    parts = [f"<div><span class='label'>type</span>{html.escape(str(answer.get('type')))}</div>"]
    if answer.get("text"):
        parts.append(f"<div><span class='label'>text</span>{html.escape(str(answer['text']))}</div>")
    if answer.get("choice"):
        parts.append(f"<div><span class='label'>choice</span>{html.escape(str(answer['choice']))}</div>")
    if answer.get("bbox"):
        parts.append(f"<div><span class='label'>bbox</span>{html.escape(str(answer['bbox']))}</div>")
    if answer.get("structured") is not None:
        text = json.dumps(answer["structured"], ensure_ascii=False)[:900]
        parts.append(f"<div><span class='label'>structured</span><pre>{html.escape(text)}</pre></div>")
    return "<div class='answer'>" + "".join(parts) + "</div>"


def render_media(record):
    media = record.get("media") or []
    if not media:
        filename = record.get("metadata", {}).get("filename")
        extra = f"<div class='meta-inline'>filename: {html.escape(str(filename))}</div>" if filename else ""
        return f"<div class='no-media'>No local media linked{extra}</div>"
    bbox = record.get("answer", {}).get("bbox")
    blocks = []
    for item in media:
        src = rel_media_src(item["path"])
        caption = html.escape(item["path"])
        if src is None:
            blocks.append(f"<div class='no-media'>Missing file: {caption}</div>")
            continue
        overlay = ""
        if bbox and len(bbox) == 4 and item.get("width") and item.get("height"):
            x, y, w, h = bbox
            left = max(0.0, min(100.0, x / item["width"] * 100.0))
            top = max(0.0, min(100.0, y / item["height"] * 100.0))
            width = max(0.0, min(100.0, w / item["width"] * 100.0))
            height = max(0.0, min(100.0, h / item["height"] * 100.0))
            overlay = (
                f"<div class='bbox' style='left:{left:.4f}%;top:{top:.4f}%;"
                f"width:{width:.4f}%;height:{height:.4f}%;'></div>"
            )
        blocks.append(
            "<figure>"
            "<div class='img-wrap'>"
            f"<img src='{html.escape(src)}' loading='lazy' />"
            f"{overlay}"
            "</div>"
            f"<figcaption>{caption}</figcaption>"
            "</figure>"
        )
    return "<div class='media-grid'>" + "".join(blocks) + "</div>"


def render_record(record, idx):
    prompt_text = html.escape(record["prompt"]["text"])
    subset = record.get("subset")
    meta_bits = [
        f"<span>{html.escape(record['benchmark'])}</span>",
        f"<span>{html.escape(record['split'])}</span>",
        f"<span>{html.escape(record['task_type'])}</span>",
    ]
    if subset:
        meta_bits.append(f"<span>{html.escape(str(subset))}</span>")
    return (
        "<article class='card'>"
        f"<div class='card-head'><h3>Sample {idx + 1}</h3><div class='badges'>{''.join(f'<span>{x}</span>' for x in meta_bits)}</div></div>"
        f"<div class='uid'>{html.escape(record['uid'])}</div>"
        f"{render_media(record)}"
        f"<div class='sec'><div class='sec-title'>Prompt</div><div class='prompt'>{prompt_text}</div></div>"
        f"{render_choices(record['prompt'])}"
        f"<div class='sec'><div class='sec-title'>Answer</div>{render_answer(record['answer'])}</div>"
        "</article>"
    )


def render_section(name, rows):
    if not rows:
        return (
            "<section class='bench'>"
            f"<h2>{html.escape(name)}</h2>"
            "<div class='cards'><article class='card'>No normalized data found yet.</article></div>"
            "</section>"
        )
    return (
        "<section class='bench'>"
        f"<h2>{html.escape(name)}</h2>"
        "<div class='cards'>"
        + "".join(render_record(row, i) for i, row in enumerate(rows))
        + "</div></section>"
    )


def main():
    sections = []
    for name, path in FILES:
        rows = load_two(path)
        sections.append(render_section(name, rows))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    shortlist_html = markdown.markdown(
        SHORTLIST_MD.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code"],
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark Preview</title>
  <style>
    :root {{
      --bg: #f6f3ee;
      --card: #fffdfa;
      --ink: #1c1a18;
      --muted: #66615a;
      --line: #d8d0c6;
      --accent: #146356;
      --accent-2: #f0e6d6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #efe6db 0, transparent 28%),
        linear-gradient(180deg, #faf8f4 0%, var(--bg) 100%);
    }}
    .wrap {{
      width: min(1440px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 28px 0 56px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 40px;
      line-height: 1;
    }}
    .intro {{
      color: var(--muted);
      margin-bottom: 26px;
      font-size: 15px;
    }}
    .bench {{
      margin: 28px 0 40px;
    }}
    .shortlist {{
      background: rgba(255,253,250,0.92);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 24px 26px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.05);
      margin: 12px 0 34px;
    }}
    .shortlist h1,
    .shortlist h2,
    .shortlist h3 {{
      margin-top: 0;
      color: var(--ink);
    }}
    .shortlist table {{
      width: 100%;
      border-collapse: collapse;
      margin: 12px 0 18px;
      font-size: 14px;
    }}
    .shortlist th,
    .shortlist td {{
      border: 1px solid var(--line);
      padding: 10px 12px;
      vertical-align: top;
      text-align: left;
    }}
    .shortlist th {{
      background: #f4ede2;
    }}
    .shortlist code {{
      background: #f3efe8;
      padding: 2px 6px;
      border-radius: 6px;
      font-size: 0.95em;
    }}
    .shortlist ul {{
      padding-left: 22px;
    }}
    .shortlist li {{
      margin: 6px 0;
      line-height: 1.5;
    }}
    .bench h2 {{
      margin: 0 0 14px;
      font-size: 24px;
      border-left: 6px solid var(--accent);
      padding-left: 12px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 12px 30px rgba(0,0,0,0.05);
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 8px;
    }}
    .card h3 {{
      margin: 0;
      font-size: 18px;
    }}
    .badges {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: flex-end;
    }}
    .badges span {{
      font-size: 12px;
      color: var(--accent);
      border: 1px solid #b8d2cd;
      background: #edf7f4;
      padding: 4px 8px;
      border-radius: 999px;
    }}
    .uid {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 10px;
      word-break: break-all;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 10px;
      margin-bottom: 12px;
    }}
    figure {{
      margin: 0;
      background: #f3ede5;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid var(--line);
    }}
    .img-wrap {{
      position: relative;
    }}
    img {{
      width: 100%;
      display: block;
      background: #ece5da;
      max-height: 420px;
      object-fit: contain;
    }}
    .bbox {{
      position: absolute;
      border: 3px solid #ff3b30;
      box-shadow: 0 0 0 9999px rgba(255, 59, 48, 0.08);
      border-radius: 4px;
      pointer-events: none;
    }}
    figcaption {{
      font-size: 12px;
      color: var(--muted);
      padding: 8px 10px;
      word-break: break-all;
    }}
    .no-media {{
      border: 1px dashed #c5b8a5;
      background: var(--accent-2);
      color: #6f5d44;
      border-radius: 14px;
      padding: 14px;
      font-size: 13px;
      margin-bottom: 12px;
    }}
    .meta-inline {{
      margin-top: 6px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      word-break: break-all;
    }}
    .sec {{
      margin-top: 10px;
    }}
    .sec-title {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .prompt {{
      white-space: pre-wrap;
      line-height: 1.5;
    }}
    .choices ol {{
      margin: 0;
      padding-left: 20px;
    }}
    .choices li {{
      margin: 4px 0;
      line-height: 1.45;
    }}
    .answer > div {{
      margin: 6px 0;
      line-height: 1.45;
    }}
    .label {{
      display: inline-block;
      min-width: 86px;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      vertical-align: top;
    }}
    pre {{
      white-space: pre-wrap;
      margin: 8px 0 0;
      padding: 12px;
      background: #f7f4ef;
      border-radius: 12px;
      border: 1px solid var(--line);
      max-height: 220px;
      overflow: auto;
      font-size: 12px;
    }}
    @media (max-width: 720px) {{
      .wrap {{ width: min(100vw - 20px, 1440px); }}
      .cards {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 30px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Benchmark Preview</h1>
    <div class="intro">Generated at {generated_at}. Each benchmark shows 2 samples from the current normalized data. Missing-media cards are intentional only when the chosen sample truly has no local image.</div>
    <section class="shortlist">
      {shortlist_html}
    </section>
    {''.join(sections)}
  </div>
</body>
</html>
"""
    OUTPUT.write_text(page, encoding="utf-8")
    stamped = ROOT / f"benchmark_preview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    stamped.write_text(page, encoding="utf-8")
    print(stamped)


if __name__ == "__main__":
    main()
