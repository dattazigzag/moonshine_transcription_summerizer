#!/usr/bin/env python3
"""
Agent 1: Transcript Cleanup
Reads a parsed markdown transcript, sends it to a local Ollama model to
remove filler words and fix transcription errors, and outputs a cleaned markdown file.
"""

import argparse
import sys
from pathlib import Path
from ollama import Client, ResponseError

# --- Global Configurations ---
# Change to "qwen2.5:27b" or your specific tag
DEFAULT_MODEL = "gemma2:27b"
# Notice that for the official client, you only provide the base host URL, not the /api/chat endpoint
DEFAULT_HOST = "http://192.168.178.160:11434"

SYSTEM_PROMPT = """
You are an expert transcript editor. Your task is to clean up a raw meeting transcript.
Strict rules to follow:
1. Remove filler words (e.g., 'um', 'ah', 'like', 'you know').
2. Fix obvious transcription errors and stuttering/false starts.
3. Do NOT summarize or remove actual context. Maintain the chronological flow and all details.
4. Maintain the exact markdown formatting provided: **Speaker Name:** [Cleaned Text]
5. Output ONLY the cleaned transcript. Do not include any introductory or concluding remarks.
"""


def clean_transcript(input_md: Path, out_dir: Path, model: str, host: str) -> Path:
    print(f"Reading {input_md.name}...")
    content = input_md.read_text(encoding="utf-8")

    # Initialize the Ollama client with your custom network URL
    client = Client(host=host)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Here is the transcript to clean:\n\n{content}"},
    ]

    print(f"Sending to Ollama ({model}) at {host}...")
    try:
        # We pass keep_alive=-1 as a top-level parameter so it stays loaded in VRAM
        response = client.chat(model=model, messages=messages, keep_alive=-1)
    except ResponseError as e:
        print(f"Ollama API Error: {e.error}")
        sys.exit(1)
    except Exception as e:
        print(f"Connection Error: {e}")
        sys.exit(1)

    # Extract the cleaned text from the response
    cleaned_text = response.get("message", {}).get("content", "").strip()

    # Save the cleaned transcript
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_md.stem}_cleaned.md"
    out_path.write_text(cleaned_text, encoding="utf-8")

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Clean up a markdown transcript using the official Ollama Python library."
    )
    parser.add_argument("input_md", type=Path, help="Input .md transcript file")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("."), help="Output directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Ollama model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help=f"Ollama host URL (default: {DEFAULT_HOST})",
    )

    args = parser.parse_args()

    if not args.input_md.exists():
        print(f"File not found: {args.input_md}")
        sys.exit(1)

    out_file = clean_transcript(args.input_md, args.out_dir, args.model, args.host)
    print(f"Successfully wrote cleaned transcript to: {out_file}")
    print("Model remains loaded in Ollama RAM for next steps.")


if __name__ == "__main__":
    main()
