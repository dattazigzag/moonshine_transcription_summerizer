#!/usr/bin/env python3
"""
Convert a moonshine.ai RTF transcript into JSON + Markdown.

Usage:
    uv run step1_convert.py input.rtf [--out-dir .]

Dependencies:
    striprtf  (pure-python, no native deps)

Output:
    <stem>.json  — structured turns, suitable for programmatic use
    <stem>.md    — human-readable, optimized for LLM summarization input
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from striprtf.striprtf import rtf_to_text

SPEAKER_RE = re.compile(r"^\s*(Speaker\s+\d+)\s*:\s*$", re.IGNORECASE)


def parse_turns(plain_text: str) -> list[dict]:
    """Group consecutive lines under each `Speaker N:` header into one turn."""
    turns: list[dict] = []
    current_speaker: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current_speaker and buffer:
            text = " ".join(s.strip() for s in buffer if s.strip())
            if text:
                turns.append(
                    {
                        "index": len(turns),
                        "speaker": current_speaker,
                        "text": text,
                    }
                )

    for raw in plain_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = SPEAKER_RE.match(line)
        if m:
            flush()
            current_speaker = m.group(1).title()  # normalize "speaker 1" -> "Speaker 1"
            buffer = []
        else:
            buffer.append(line)
    flush()
    return turns


def build_markdown(turns: list[dict], source_name: str) -> str:
    lines = [f"# Transcript: {source_name}", ""]
    for t in turns:
        lines.append(f"**{t['speaker']}:** {t['text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_json(turns: list[dict], source_name: str) -> dict:
    speaker_counts = Counter(t["speaker"] for t in turns)
    word_count = sum(len(t["text"].split()) for t in turns)
    return {
        "source": source_name,
        "n_turns": len(turns),
        "speakers": sorted(speaker_counts.keys(), 
                        key=lambda s: int(s.split()[-1])),
        "turns_per_speaker": dict(speaker_counts),
        "word_count": word_count,
        "turns": turns,
    }


def convert(rtf_path: Path, out_dir: Path) -> tuple[Path, Path]:
    rtf_text = rtf_path.read_text(encoding="utf-8", errors="replace")
    plain = rtf_to_text(rtf_text, errors="ignore")
    turns = parse_turns(plain)
    if not turns:
        raise SystemExit(
            f"No speaker turns found in {rtf_path}. " "Is this really a moonshine RTF?"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{rtf_path.stem}.json"
    md_path = out_dir / f"{rtf_path.stem}.md"

    json_path.write_text(
        json.dumps(build_json(turns, rtf_path.name), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(build_markdown(turns, rtf_path.name), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    doc_lines = (__doc__ or "").splitlines()
    ap = argparse.ArgumentParser(
        description=doc_lines[1] if len(doc_lines) > 1 else ""
    )
    ap.add_argument("rtf", type=Path, help="Input .rtf transcript")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: cwd)",
    )
    args = ap.parse_args()

    json_path, md_path = convert(args.rtf, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
