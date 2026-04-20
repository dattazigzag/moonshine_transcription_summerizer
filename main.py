#!/usr/bin/env python3
"""
Master Orchestrator for the Meeting Summarization Pipeline.
Processes an RTF transcript through cleanup, mapping, extraction, and formatting.
"""

import argparse
import sys
from pathlib import Path

# Import the core functions from our pipeline modules
from pipeline.step1_convert import convert
from pipeline.step2_cleanup import clean_transcript
from pipeline.step3_mapping import map_speakers
from pipeline.step4_extraction import extract_information
from pipeline.step5_formatter import format_summary


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Local LLM Meeting Summarizer."
    )

    # Required Input
    parser.add_argument("input_rtf", type=Path, help="Path to the raw .rtf transcript")

    # Global/Path Options
    parser.add_argument(
        "--out-dir", type=Path, default=Path("./output"), help="Base output directory"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="http://192.168.178.160:11434",
        help="Ollama host URL",
    )

    # Model Selection Options (Allowing the Mix-and-Match strategy)
    parser.add_argument(
        "--cleanup-model",
        type=str,
        default="gemma4:26b",
        help="Model for Step 2: Cleanup",
    )
    parser.add_argument(
        "--extract-model",
        type=str,
        default="gemma4:26b",
        help="Model for Step 4: Extraction",
    )
    parser.add_argument(
        "--format-model",
        type=str,
        default="gemma4:26b",
        help="Model for Step 5: Formatting",
    )

    args = parser.parse_args()

    if not args.input_rtf.exists():
        print(f"Error: Input file not found: {args.input_rtf}")
        sys.exit(1)

    print(f"\n Starting Pipeline for: {args.input_rtf.name}")
    print("-" * 40)

    # --- Setup Directories ---
    dir_raw = args.out_dir / "raw_files"
    dir_cleaned = args.out_dir / "cleaned_files"
    dir_named = args.out_dir / "named_files"
    dir_extracted = args.out_dir / "extracted_files"
    dir_final = args.out_dir / "final_summaries"

    # --- Execute Pipeline ---
    try:
        # Step 1: Parse RTF
        print("\n[1/5] Parsing RTF to Markdown...")
        _, md_path = convert(args.input_rtf, dir_raw)

        # Step 2: Cleanup
        print(f"\n[2/5] Cleaning transcript using {args.cleanup_model}...")
        cleaned_path = clean_transcript(
            md_path, dir_cleaned, args.cleanup_model, args.host
        )

        # Step 3: Human-in-the-Loop Mapping
        print("\n[3/5] Human-in-the-loop Speaker Mapping...")
        named_path = map_speakers(cleaned_path, dir_named)

        # Step 4: Extraction
        print(f"\n[4/5] Extracting intelligence using {args.extract_model}...")
        extracted_path = extract_information(
            named_path, dir_extracted, args.extract_model, args.host
        )

        # Step 5: Formatting
        print(f"\n[5/5] Formatting final summary using {args.format_model}...")
        final_path = format_summary(
            extracted_path, dir_final, args.format_model, args.host
        )

        print("-" * 40)
        print(f"✅ Pipeline Complete! Final summary saved to:\n{final_path.absolute()}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Pipeline interrupted by user. Exiting.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
