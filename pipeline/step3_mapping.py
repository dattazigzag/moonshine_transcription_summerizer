#!/usr/bin/env python3
"""
Step 3: Speaker Mapping (Human-in-the-Loop)
Finds generic speaker tags in a cleaned transcript, prompts the user
for their real names, and outputs a newly named markdown file.

The detection and application primitives (`detect_generic_speakers`,
`apply_speaker_mapping`) are exposed separately so the Gradio web UI
can reuse them without going through the blocking CLI `input()` loop.
"""

import argparse
import re
import sys
from pathlib import Path


# Matches '**Speaker N:**' tags (case-insensitive). Module-level so both
# the CLI wrapper and the Gradio path share one definition.
SPEAKER_PATTERN = re.compile(r"\*\*(Speaker\s+\d+):\*\*", re.IGNORECASE)


def is_valid_name(name: str) -> bool:
    """Guardrail: Ensure the name only contains letters, spaces, hyphens, or apostrophes."""
    if not name:  # Allow empty (user skips)
        return True
    return bool(re.match(r"^[A-Za-z\s\-'.]+$", name))


def detect_generic_speakers(content: str) -> list[str]:
    """Return the sorted unique 'Speaker N' tags found in content.

    Pure function. No I/O. Used by both the CLI path and the Gradio path
    to decide whether a speaker-naming prompt is needed.

        >>> detect_generic_speakers("**Speaker 2:** hi\\n**Speaker 1:** bye")
        ['Speaker 1', 'Speaker 2']
        >>> detect_generic_speakers("**Amanda:** hi")
        []
    """
    return sorted(set(SPEAKER_PATTERN.findall(content)))


def apply_speaker_mapping(content: str, mapping: dict[str, str]) -> str:
    """Return content with '**{old}:**' replaced by '**{new}:**' per mapping.

    Pure function. No I/O.

    Rules:
      * Keys whose value is an empty string are skipped (original tag
        preserved). This lets callers pass a full mapping dict without
        pre-filtering — blank inputs are treated as "leave as-is".
      * Keys that never appear in content are no-ops.
      * Substitution is case-insensitive, matching the detect regex.

        >>> apply_speaker_mapping("**Speaker 1:** hi", {"Speaker 1": "Alice"})
        '**Alice:** hi'
        >>> apply_speaker_mapping("**Speaker 1:** hi", {"Speaker 1": ""})
        '**Speaker 1:** hi'
        >>> apply_speaker_mapping("**Speaker 1:** hi", {"Speaker 9": "Bob"})
        '**Speaker 1:** hi'
    """
    for old_tag, new_name in mapping.items():
        if not new_name:
            continue
        content = re.sub(
            rf"\*\*{re.escape(old_tag)}:\*\*",
            f"**{new_name}:**",
            content,
            flags=re.IGNORECASE,
        )
    return content


def map_speakers(input_md: Path, out_dir: Path) -> Path:
    """CLI entry point: read, detect, prompt, apply, write.

    Behavior MUST stay bit-identical to the pre-refactor version — verified
    by M1 regression diff against `output/named_files/*_named.md`.
    """
    print(f"Reading {input_md.name}...")
    content = input_md.read_text(encoding="utf-8")

    unique_speakers = detect_generic_speakers(content)

    if not unique_speakers:
        print("No generic 'Speaker X' tags found. Moving on.")
        out_path = out_dir / f"{input_md.stem.replace('_cleaned', '')}_named.md"
        out_path.write_text(content, encoding="utf-8")
        return out_path

    print(f"\nFound {len(unique_speakers)} generic speakers. Let's map them.")
    print("Leave blank and press Enter to keep the original label.\n")

    replacements: dict[str, str] = {}
    for speaker in unique_speakers:
        while True:
            new_name = input(f"Enter real name for '{speaker}': ").strip()
            if is_valid_name(new_name):
                break
            print(
                "Invalid input. Please use only letters, spaces, hyphens, or apostrophes."
            )

        if new_name:
            replacements[speaker] = new_name

    content = apply_speaker_mapping(content, replacements)

    out_dir.mkdir(parents=True, exist_ok=True)
    new_stem = input_md.stem.replace("_cleaned", "")
    out_path = out_dir / f"{new_stem}_named.md"

    out_path.write_text(content, encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Map generic speaker labels to real names."
    )
    parser.add_argument("input_md", type=Path, help="Input cleaned .md transcript file")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."), help="Output directory"
    )

    args = parser.parse_args()

    if not args.input_md.exists():
        print(f"File not found: {args.input_md}")
        sys.exit(1)

    out_file = map_speakers(args.input_md, args.out_dir)
    print(f"\nSuccessfully wrote named transcript to: {out_file}")


if __name__ == "__main__":
    main()
