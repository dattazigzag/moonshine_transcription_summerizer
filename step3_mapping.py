#!/usr/bin/env python3
"""
Step 3: Speaker Mapping (Human-in-the-Loop)
Finds generic speaker tags in a cleaned transcript, prompts the user
for their real names, and outputs a newly named markdown file.
"""

import argparse
import re
import sys
from pathlib import Path


def is_valid_name(name: str) -> bool:
    """Guardrail: Ensure the name only contains letters, spaces, hyphens, or apostrophes."""
    if not name:  # Allow empty (user skips)
        return True
    return bool(re.match(r"^[A-Za-z\s\-'.]+$", name))


def map_speakers(input_md: Path, out_dir: Path) -> Path:
    print(f"Reading {input_md.name}...")
    content = input_md.read_text(encoding="utf-8")

    # Find all unique "Speaker X" tags.
    # Our previous script formatted them as **Speaker X:**
    speaker_pattern = re.compile(r"\*\*(Speaker\s+\d+):\*\*", re.IGNORECASE)
    unique_speakers = sorted(set(speaker_pattern.findall(content)))

    if not unique_speakers:
        print("No generic 'Speaker X' tags found. Moving on.")
        out_path = out_dir / f"{input_md.stem.replace('_cleaned', '')}_named.md"
        out_path.write_text(content, encoding="utf-8")
        return out_path

    print(f"\nFound {len(unique_speakers)} generic speakers. Let's map them.")
    print("Leave blank and press Enter to keep the original label.\n")

    replacements = {}
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

    # Replace the generic tags with the new names in the content
    for old_tag, new_name in replacements.items():
        # Replace Speaker x: with <name>:
        content = re.sub(
            rf"\*\*{old_tag}:\*\*", f"**{new_name}:**", content, flags=re.IGNORECASE
        )

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
