#!/usr/bin/env python3
"""
Gradio front-end for the Local Meeting Transcript Summarizer.

Thin skin over ``main.py``'s pipeline. See ``contexts/gradio_app.md`` for the
full spec, decisions, and milestone breakdown.

Run:
    uv run app.py

Then open http://localhost:7860 in a browser.
"""

from __future__ import annotations

import argparse
import atexit
import base64
import contextlib
import io
import os
import re
import shutil
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO, cast

import gradio as gr
import httpx
from dotenv import load_dotenv

from pipeline import (
    announce,
    announce_done,
    announce_start,
    announce_unload,
    announce_unload_result,
)
from pipeline.pdf_export import md_to_pdf
from pipeline.step1_convert import convert
from pipeline.step2_cleanup import clean_transcript
from pipeline.step3_mapping import apply_speaker_mapping
from pipeline.step4_extraction import extract_information
from pipeline.step5_formatter import format_summary


# Load .env at module import so OLLAMA_HOST is populated BEFORE the constants
# block evaluates. main() hard-fails if it's still missing.
load_dotenv()


# ─── Defaults ─────────────────────────────────────────────────────────────

DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "")
DEFAULT_EDITOR_MODEL = "gemma4:26b"
DEFAULT_EXTRACTOR_MODEL = "gemma4:26b"

SERVER_PORT = 7860
SERVER_HOST = "0.0.0.0"

OLLAMA_PROBE_TIMEOUT = 3.0
UPLOAD_MAX_SIZE = "10mb"

# Stable intermediate-file stem used by steps 2→5 (M6.5 S3).
STABLE_STEM = "transcript"

# Log panel sizing (M6.5 L1).
LOG_PANEL_LINES = 12

# Fixed preview/summary heights. Same height on summary + raw textbox so
# the Rendered/Raw toggle doesn't jump the page.
PREVIEW_HEIGHT = 400
SUMMARY_HEIGHT = 500
SUMMARY_RAW_LINES = 20

# Polling interval for the threaded stdout streamer (below). 0.3s is the
# sweet spot between UI-refresh spam and log-line latency.
LOG_POLL_INTERVAL = 0.3

# Refresh-button icon, relative to app.py. Gradio's Button.icon= serves
# files via its static-file cache, so a relative path is enough.
REFRESH_ICON_PATH = "assets/refresh.svg"


# ─── Speaker detection regex ─────────────────────────────────────────────

_SPEAKER_LINE_RE = re.compile(r"^\*\*([^:]+):\*\*")
_GENERIC_SPEAKER_RE = re.compile(r"^Speaker \d+$")


def detect_all_speakers(md_text: str) -> list[tuple[str, bool]]:
    """Return [(speaker_name, is_generic), ...] preserving first-seen order.

    ``is_generic`` is True when the name matches 'Speaker N'. The UI uses
    this to decide whether to render each textbox as editable (generic,
    user needs to fill in) or disabled (already named, just display).
    """
    seen: dict[str, bool] = {}
    for line in md_text.splitlines():
        m = _SPEAKER_LINE_RE.match(line)
        if not m:
            continue
        name = m.group(1).strip()
        if name not in seen:
            seen[name] = bool(_GENERIC_SPEAKER_RE.match(name))
    return list(seen.items())


# ─── Process-level tracking for shutdown hooks ────────────────────────────

_ALL_MODELS_EVER_LOADED: set[tuple[str, str]] = set()


# ─── Custom CSS ───────────────────────────────────────────────────────────

# Injected via demo.launch(css=...) — Gradio 6 moved this off the Blocks
# constructor. Three blocks:
#
#   .log-panel textarea — monospace font for the log panel.
#   #conn-indicator / #sidebar-host-row — dot-indicator alignment in the
#       sidebar.
#   #progress-label ... — bumps the contrast of gr.Label's internal bar.
#       Selectors target likely internal classes (``[class*="confidence"]``
#       / ``[class*="fill"]``) with broad ``!important`` overrides so we
#       don't depend on Gradio's build-hashed class names. Works even if
#       Gradio's internal structure shifts minor versions.
#   #summary-container + children — CSS-class-driven Rendered/Raw toggle.
#       Both gr.Markdown and gr.Textbox stay ``visible=True`` in Python;
#       which one is displayed depends on whether the outer column has
#       ``.view-raw``. Avoids Gradio 6's buggy ``visible="hidden"`` path
#       (which lets the raw textbox leak through on some code paths).
CUSTOM_CSS = """
.log-panel textarea {
    font-family: var(--font-mono, 'JetBrains Mono', ui-monospace, 'SF Mono', 'Cascadia Code', Menlo, monospace) !important;
    font-size: 0.82em !important;
    line-height: 1.45 !important;
}

#conn-indicator {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    min-height: 36px;
    padding-left: 6px;
}

#sidebar-host-row {
    gap: 8px;
    align-items: center;
}

/* Progress bar contrast (gr.Label).
   gr.Label renders each confidence row as a native <meter class="bar">
   element. To restyle it we need appearance:none + the WebKit/Moz meter
   pseudo-elements; plain class selectors don't reach the fill. */
#progress-label {
    background: var(--neutral-900, #0f0f0f) !important;
    padding: 10px 14px !important;
    border-radius: 6px !important;
    border: 1px solid var(--border-color-primary, rgba(255, 255, 255, 0.08)) !important;
}
#progress-label meter {
    -webkit-appearance: none !important;
    appearance: none !important;
    width: 100% !important;
    height: 10px !important;
    border: none !important;
    border-radius: 5px !important;
    background: rgba(255, 255, 255, 0.08) !important;
}
#progress-label meter::-webkit-meter-bar {
    background: rgba(255, 255, 255, 0.08) !important;
    border: none !important;
    border-radius: 5px !important;
}
#progress-label meter::-webkit-meter-optimum-value,
#progress-label meter::-webkit-meter-suboptimum-value,
#progress-label meter::-webkit-meter-even-less-good-value {
    background: #11ba88 !important;
    border-radius: 5px !important;
    transition: width 0.25s ease;
}
#progress-label meter::-moz-meter-bar {
    background: #11ba88 !important;
    border-radius: 5px !important;
}

/* Rendered/Raw toggle — display controlled by a class on the outer column
   (elem_id="summary-container"). Default: only the rendered markdown
   shows. When .view-raw is applied, the raw textbox shows and the
   rendered markdown hides. M11 adds .view-pdf as a third state: the
   iframe shows, rendered + raw hide. Classes are mutually exclusive
   (the JS toggle clears both before setting one). */
#final-summary-source { display: none !important; }
#summary-container.view-raw #final-summary-source { display: block !important; }
#summary-container.view-raw #final-summary { display: none !important; }

/* PDF mode (M11) — iframe shown, rendered markdown + raw textbox hidden. */
#final-summary-pdf { display: none !important; }
#summary-container.view-pdf #final-summary-pdf { display: block !important; }
#summary-container.view-pdf #final-summary { display: none !important; }
#summary-container.view-pdf #final-summary-source { display: none !important; }

/* Refresh button: icon-only (no label text), sized to match inputs. */
#test-btn button {
    display: flex;
    align-items: center;
    justify-content: center;
    min-width: 40px;
    padding: 6px !important;
}
#test-btn button img {
    width: 16px;
    height: 16px;
    margin: 0 !important;
}

/* App footer — attribution line that sits at the bottom of our main
   gr.Column, which means it renders immediately above Gradio's own
   built-in "Use via API · Built with Gradio · Settings" strip.
   (Gradio 6 has no public API to inject items INTO that strip, so
   our line + Gradio's line are stacked siblings at the page end.)

   Sticky-footer trick: on short pages (empty state, no upload yet)
   we want the footer to land near the bottom, not float in the
   middle of a blank page. Done here with flexbox on our main column:
   give it min-height that fills the viewport minus a rough Gradio
   footer reservation, make it a flex column, and push #app-footer
   down with margin-top: auto. On long pages (after a summary has
   rendered) the column grows past min-height and the footer sits
   directly after the Download button — exactly the natural-flow
   behaviour we want. */
.gradio-container > .main > .contain > .wrap,
.gradio-container > .main > .contain,
.gradio-container > .main {
    min-height: calc(100vh - 60px);  /* 60px ≈ Gradio's own footer */
}

/* Target the main content column (the gr.Column sibling to the
   sidebar) and make it a flex container so margin-top: auto on the
   footer works. Gradio wraps columns in a div with class `.column`
   inside a `.form`/`.panel`; we rely on the common `.column` class. */
#app-footer {
    margin-top: auto;       /* pushes to bottom when column has extra height */
    padding-top: 24px;
    padding-bottom: 8px;
    text-align: center;
    opacity: 0.55;
    font-size: 0.85em;
}
#app-footer a {
    color: inherit;
    text-decoration: none;
}
#app-footer a:hover {
    opacity: 1;
    text-decoration: underline;
}
"""


# ─── Dark-mode forcing ────────────────────────────────────────────────────
FORCE_DARK_MODE_JS = """
() => {
    if (!document.body.classList.contains('dark')) {
        document.body.classList.add('dark');
    }
}
"""

# Radio toggle: pure JS, no Python roundtrip. Flips mutually-exclusive
# CSS classes on the outer column (.view-raw / .view-pdf). Default
# state = neither class = rendered markdown shows.
TOGGLE_VIEW_MODE_JS = """
(mode) => {
    const container = document.getElementById('summary-container');
    if (!container) return;
    container.classList.remove('view-raw', 'view-pdf');
    if (mode === 'Raw') {
        container.classList.add('view-raw');
    } else if (mode === 'PDF') {
        container.classList.add('view-pdf');
    }
}
"""

# Fires whenever the rendered markdown's value changes (success yield sets
# it to the final content, reset/error yields set it to ""). Resets the
# container to rendered mode AND programmatically clicks the Rendered
# radio so its visible state matches the CSS (Gradio's radio .change()
# doesn't fire on gr.update(value=...); the click() is the workaround).
# M11: clears both .view-raw AND .view-pdf so a new run starts clean.
RESET_VIEW_MODE_JS = """
() => {
    const container = document.getElementById('summary-container');
    if (container) container.classList.remove('view-raw', 'view-pdf');
    const renderedInput = document.querySelector('#view-mode input[value="Rendered"]');
    if (renderedInput && !renderedInput.checked) renderedInput.click();
}
"""

# JS fired by the Copy button. Reads raw markdown source from the hidden
# gr.Textbox's <textarea> (always in DOM even when CSS-hidden via
# display:none).
COPY_SUMMARY_JS = """
() => {
    const host = document.getElementById('final-summary-source');
    if (!host) return;
    const ta = host.querySelector('textarea');
    if (!ta) return;
    navigator.clipboard.writeText(ta.value || '');
}
"""


# ─── Threaded log streaming ──────────────────────────────────────────────


class _Tee(io.TextIOBase):
    """Write-only stream that fans out to multiple underlying streams.

    Used so step prints land in BOTH our polling buffer (for the UI log
    panel) AND the original terminal stdout (so ``uv run app.py`` still
    shows live progress in the console). A bare
    ``contextlib.redirect_stdout(buf)`` would silence the terminal, which
    is the wrong trade-off for a dev tool.
    """

    def __init__(self, *streams: Any) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                # A single sink failing (e.g. closed terminal) must not
                # kill the step. Skip it, keep writing to the others.
                pass
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass


def _stream_step(
    state_val: dict[str, Any],
    result_key: str,
    fn: Callable[[], Any],
) -> Iterator[None]:
    """Run ``fn`` in a daemon thread with stdout captured. Yield each time
    the captured output grows.

    Mutates ``state_val``:
        * ``state_val['log_text']`` — appended to with each poll's chunk
        (initial log preserved so prior steps' output stays visible).
        * ``state_val[result_key]`` — set to ``fn``'s return value on
        successful completion.

    Exceptions inside ``fn`` are re-raised from this generator when it
    exits, so callers can wrap the iteration in try/except and see the
    error normally. ``SystemExit`` (pipeline's ``sys.exit(1)``) is caught
    too, since it's a BaseException subclass.

    Caveat: ``contextlib.redirect_stdout`` patches sys.stdout globally,
    so any writes from any thread during the redirect land in the
    buffer. Fine for single-user usage; multi-user concurrent runs would
    interleave captures. Revisit if that's ever a real scenario.

    Cancellation (Gradio's ``cancels=`` raising GeneratorExit) interrupts
    the polling loop but NOT the step thread. The step's Ollama call
    continues in the background until it returns, then the daemon thread
    exits silently. Models unload in the outer try/finally.

    Plain ``io.StringIO`` under the GIL is relied on for buffer safety —
    CPython serialises the underlying list-of-chunks ops. A previous
    round used a lock-wrapped class; the lock adds no real safety over
    GIL here, so it was removed.
    """
    buf = io.StringIO()
    result: list[Any] = [None]
    err: list[BaseException | None] = [None]

    # Capture the real stdout now, BEFORE redirect_stdout swaps it out
    # globally inside the worker. Without this snapshot, the _Tee below
    # would write to the already-replaced stdout and the terminal would
    # still go silent.
    original_stdout = sys.stdout

    def _worker() -> None:
        try:
            tee = _Tee(buf, original_stdout)
            with contextlib.redirect_stdout(cast(TextIO, tee)):
                result[0] = fn()
        except BaseException as e:
            err[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    initial = state_val.get("log_text", "")
    last_len = 0

    while t.is_alive():
        time.sleep(LOG_POLL_INTERVAL)
        current = buf.getvalue()
        if len(current) > last_len:
            state_val["log_text"] = initial + current
            yield
            last_len = len(current)

    t.join(timeout=1.0)
    final = buf.getvalue()
    if len(final) != last_len:
        state_val["log_text"] = initial + final
        yield

    if err[0] is not None:
        raise err[0]
    state_val[result_key] = result[0]


# ─── Progress-bar helper ─────────────────────────────────────────────────


def _progress_value(phase: str, pct: int) -> dict[str, float]:
    """Shape the value dict gr.Label expects: {label: fraction}.

    One entry = one bar. gr.Label renders the key as the bar's label and
    the value (0.0–1.0) as the fill width, with the percentage on the
    right. Theme accent colour + our CSS overrides for track/fill
    contrast handle the rest.
    """
    try:
        pct_int = max(0, min(100, int(pct)))
    except (TypeError, ValueError):
        pct_int = 0
    return {phase: pct_int / 100.0}


# ─── Ollama helpers ───────────────────────────────────────────────────────


def test_ollama_connection(host: str) -> tuple[bool, str]:
    if not host or not host.strip():
        return False, "Host is empty."
    url = f"{host.rstrip('/')}/api/tags"
    try:
        r = httpx.get(url, timeout=OLLAMA_PROBE_TIMEOUT)
        r.raise_for_status()
        return True, f"✓ Connected to {host}"
    except httpx.ConnectError:
        return False, f"✗ Cannot reach {host} (connection refused)"
    except httpx.TimeoutException:
        return False, f"✗ Cannot reach {host} (timed out after {OLLAMA_PROBE_TIMEOUT}s)"
    except httpx.HTTPStatusError as e:
        return False, f"✗ {host} responded with HTTP {e.response.status_code}"
    except Exception as e:
        return False, f"✗ {host}: {type(e).__name__}: {e}"


def list_available_models(host: str) -> list[str]:
    if not host or not host.strip():
        return []
    url = f"{host.rstrip('/')}/api/tags"
    try:
        r = httpx.get(url, timeout=OLLAMA_PROBE_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def validate_model_available(host: str, model: str) -> bool:
    if not model or not host:
        return False
    return model in list_available_models(host)


def unload_model(host: str, model: str) -> None:
    if not host or not model:
        return
    url = f"{host.rstrip('/')}/api/generate"
    try:
        httpx.post(
            url,
            json={"model": model, "keep_alive": 0, "prompt": ""},
            timeout=OLLAMA_PROBE_TIMEOUT,
        )
    except Exception:
        pass


def preflight_check(
    host: str, editor_model: str, extractor_model: str
) -> tuple[bool, str]:
    ok, msg = test_ollama_connection(host)
    if not ok:
        return False, msg
    models = set(list_available_models(host))
    missing = [m for m in (editor_model, extractor_model) if m and m not in models]
    if missing:
        return False, f"✗ Models not pulled on {host}: {', '.join(missing)}"
    return True, "✓ Ready"


# ─── Process-level shutdown handlers ─────────────────────────────────────


def _global_cleanup_loaded_models() -> None:
    for host, model in list(_ALL_MODELS_EVER_LOADED):
        unload_model(host, model)


def _sigterm_handler(signum: int, frame: Any) -> None:
    _global_cleanup_loaded_models()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _install_process_hooks() -> None:
    atexit.register(_global_cleanup_loaded_models)
    try:
        signal.signal(signal.SIGTERM, _sigterm_handler)
    except (ValueError, OSError):
        print(
            "Warning: could not install SIGTERM handler; "
            "model cleanup on container stop will be best-effort.",
            file=sys.stderr,
        )


# ─── Session state factory ───────────────────────────────────────────────


def init_session_state() -> dict[str, Any]:
    """Factory for ``gr.State``. See contexts/gradio_app.md for full shape."""
    return {
        "tempdir_path": None,
        "canonical_md": None,
        "uploaded_stem": None,
        "final_summary_path": None,
        "final_summary_pdf_path": None,  # M11: set on first PDF-radio click, cleared on new upload / new run.
        "models_used": set(),
        "ollama_host": DEFAULT_OLLAMA_HOST,
        "run_in_progress": False,
        "log_text": "",
        "progress_pct": 0,
        "progress_phase": "",
    }


def _ensure_tempdir(state_val: dict[str, Any]) -> Path:
    if not state_val.get("tempdir_path"):
        state_val["tempdir_path"] = tempfile.mkdtemp(prefix="meeting_summarizer_")
    return Path(state_val["tempdir_path"])


def cleanup_session(state_val: dict[str, Any] | None = None) -> None:
    """Per-session cleanup via ``gr.State.delete_callback``."""
    if not isinstance(state_val, dict):
        return
    host = state_val.get("ollama_host", "")
    for model in list(state_val.get("models_used", ())):
        unload_model(host, model)
    tempdir_path = state_val.get("tempdir_path")
    if tempdir_path:
        shutil.rmtree(tempdir_path, ignore_errors=True)


# ─── UI event handlers ───────────────────────────────────────────────────


def _banner_update_for_host(host: str) -> dict:
    ok, _ = test_ollama_connection(host)
    if ok:
        return gr.update(value="", visible=False)
    return gr.update(
        value=(
            f"⚠ Cannot reach Ollama at `{host}`. "
            "Update the host in the sidebar and click the refresh button."
        ),
        visible=True,
    )


def _connection_indicator_html(host: str) -> str:
    """LED indicator as a CSS-coloured circle inside gr.HTML.

    Not an emoji — 🟢/🔴 render as squares on macOS in sans-serif contexts
    (emoji font fallback artefact). A plain coloured <div> with
    ``border-radius: 50%`` guarantees a circle everywhere.
    """
    if not host or not host.strip():
        color = "#888888"
        title = "no host configured"
    else:
        ok, _ = test_ollama_connection(host)
        color = "#11ba88" if ok else "#ef4444"
        title = "connected" if ok else "unreachable"
    return (
        f'<div title="{title}" '
        f'style="width: 12px; height: 12px; border-radius: 50%; '
        f'background: {color}; box-shadow: 0 0 4px {color}55;"></div>'
    )


def _model_indicator(host: str, model: str) -> str:
    if not model:
        return ""
    if not host:
        return "—"
    models = list_available_models(host)
    if not models:
        return "— _(host unreachable or no models pulled)_"
    if model in models:
        return "✓ available"
    return "✗ not pulled"


def on_startup(state_val: dict[str, Any]) -> tuple[dict, str, str, str]:
    host = state_val.get("ollama_host", DEFAULT_OLLAMA_HOST)
    banner = _banner_update_for_host(host)
    editor_ind = _model_indicator(host, DEFAULT_EDITOR_MODEL)
    extractor_ind = _model_indicator(host, DEFAULT_EXTRACTOR_MODEL)
    conn_html = _connection_indicator_html(host)
    return banner, editor_ind, extractor_ind, conn_html


def on_host_change(
    host: str, editor: str, extractor: str, state_val: dict[str, Any]
) -> tuple[dict, dict, str, str, str]:
    state_val["ollama_host"] = host
    banner = _banner_update_for_host(host)
    editor_ind = _model_indicator(host, editor)
    extractor_ind = _model_indicator(host, extractor)
    conn_html = _connection_indicator_html(host)
    return state_val, banner, editor_ind, extractor_ind, conn_html


def on_test_connection(host: str) -> tuple[str, dict]:
    ok, _ = test_ollama_connection(host)
    conn_html = _connection_indicator_html(host)
    if ok:
        banner = gr.update(value="", visible=False)
    else:
        banner = gr.update(
            value=(
                f"⚠ Cannot reach Ollama at `{host}`. "
                "Check the host URL and try again."
            ),
            visible=True,
        )
    return conn_html, banner


def on_stop(state_val: dict[str, Any]) -> tuple:
    """Stop-button click handler. 10-tuple (trimmed from 12 — final_summary_md
    and final_summary_source no longer need visibility updates)."""
    log_text = state_val.get("log_text", "")
    if log_text and not log_text.endswith("\n"):
        log_text += "\n"
    log_text += (
        "\n⏸ Cancel requested. Current step will finish before models "
        "unload (non-streaming Ollama trade-off; usually 1–3 min).\n"
    )
    state_val["log_text"] = log_text

    pct = state_val.get("progress_pct", 0)
    phase = state_val.get("progress_phase", "")
    header = f"⏸ Cancelled during {phase}" if phase else "⏸ Cancelled by user"

    return (
        gr.update(visible=True),  # console_group
        gr.update(value=_progress_value(header, pct)),  # progress_label
        gr.update(value=log_text),  # log_panel
        gr.update(visible=False),  # summary_section
        gr.update(visible=False),  # copy_btn
        gr.update(value=None, visible=False),  # download_btn
        gr.update(interactive=True),  # run_btn
        state_val,  # session_state
        gr.update(visible=False),  # stop_btn
    )


# 7-entry reset tuple for the seven run-output components below the
# main top-level outputs. Used by on_file_upload to clear state on new
# upload / upload clear. Order MUST match build_demo's outputs list.
_RUN_OUTPUT_RESET = (
    gr.update(visible=False),  # console_group
    gr.update(value={}),  # progress_label
    gr.update(value=""),  # log_panel
    gr.update(visible=False),  # summary_section
    gr.update(visible=False),  # copy_btn
    gr.update(value=None, visible=False),  # download_btn
)


def on_file_upload(
    uploaded_file: str | None,
    state_val: dict[str, Any],
) -> tuple:
    """Handle file upload / clear. 13-tuple return.

    Order:
        preview_md, error_md, run_btn, all_speakers_state,
        speaker_map_state, session_state, meta_md,
        + 6 entries from _RUN_OUTPUT_RESET.
    """
    if uploaded_file is None:
        state_val["canonical_md"] = None
        state_val["uploaded_stem"] = None
        state_val["final_summary_path"] = None
        state_val["final_summary_pdf_path"] = None
        state_val["log_text"] = ""
        state_val["progress_pct"] = 0
        state_val["progress_phase"] = ""
        return (
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(interactive=False),
            [],  # all_speakers_state
            {},  # speaker_map_state
            state_val,
            gr.update(value="", visible=False),  # meta_md
            *_RUN_OUTPUT_RESET,
        )

    src = Path(uploaded_file)

    try:
        tempdir = _ensure_tempdir(state_val)
        raw_dir = tempdir / "raw_files"
        raw_dir.mkdir(parents=True, exist_ok=True)
        dst = raw_dir / src.name
        shutil.copy(src, dst)
        _, md_path = convert(dst, raw_dir)
    except BaseException as e:
        err_type = type(e).__name__
        err_msg = str(e) or err_type
        state_val["canonical_md"] = None
        state_val["uploaded_stem"] = None
        state_val["final_summary_path"] = None
        state_val["final_summary_pdf_path"] = None
        state_val["log_text"] = ""
        state_val["progress_pct"] = 0
        state_val["progress_phase"] = ""
        return (
            gr.update(value="", visible=False),
            gr.update(
                value=f"❌ Could not ingest `{src.name}`: {err_msg}",
                visible=True,
            ),
            gr.update(interactive=False),
            [],
            {},
            state_val,
            gr.update(value="", visible=False),
            *_RUN_OUTPUT_RESET,
        )

    canonical_md = md_path.read_text(encoding="utf-8")
    state_val["canonical_md"] = str(md_path)
    state_val["uploaded_stem"] = src.stem
    state_val["final_summary_path"] = None
    state_val["final_summary_pdf_path"] = None
    state_val["log_text"] = ""
    state_val["progress_pct"] = 0
    state_val["progress_phase"] = ""

    all_speakers = detect_all_speakers(canonical_md)
    generic_count = sum(1 for _, is_gen in all_speakers if is_gen)

    n_turns = sum(1 for line in canonical_md.splitlines() if line.startswith("**"))
    meta_bits = [f"{n_turns} turns ingested"]
    if generic_count:
        meta_bits.append(f"{generic_count} generic speaker(s) to name")
    elif all_speakers:
        meta_bits.append(f"{len(all_speakers)} speakers — all named")
    else:
        meta_bits.append("no speakers detected")
    meta_line = " · ".join(meta_bits)

    return (
        gr.update(value=canonical_md, visible=True),
        gr.update(value="", visible=False),
        gr.update(interactive=True),
        all_speakers,
        {},
        state_val,
        gr.update(value=f"<sub>{meta_line}</sub>", visible=True),
        *_RUN_OUTPUT_RESET,
    )


# ─── Pipeline orchestration ──────────────────────────────────────────────


def run_pipeline_generator(
    state_val: dict[str, Any],
    editor_model: str,
    extractor_model: str,
    ollama_host: str,
    speaker_map: dict[str, str],
    progress: gr.Progress = gr.Progress(),
) -> Iterator[tuple]:
    """Run steps 2 → 5. 12-tuple yields.

    Tuple order (matches the outputs list in build_demo):
        0. console_group.visible
        1. progress_label.value
        2. log_panel.value
        3. summary_section.visible
        4. view_mode.value
        5. final_summary_md.value   (visibility is CSS-class driven, so
                                      only the value changes here)
        6. final_summary_source.value
        7. copy_btn.visible
        8. download_btn.value + visibility
        9. run_btn.interactive
        10. session_state
        11. stop_btn.visible

    Streaming (M6.5 round 3→4): each Ollama-touching step runs in a
    daemon thread via ``_stream_step``. The main generator polls the
    captured stdout every LOG_POLL_INTERVAL and yields a refreshed log
    panel. Pre-Ollama prints ("Reading…", "Sending to Ollama…") appear
    within ~300ms of the Run click.

    Cancellation keeps the non-streaming-Ollama delay (spec F3) — the
    main thread returns to idle immediately, the step thread's blocking
    HTTP call finishes in the background before models actually unload.
    """

    def _running_tuple(stop_visible: bool = True) -> tuple:
        return (
            gr.update(visible=True),  # console_group
            gr.update(
                value=_progress_value(
                    state_val["progress_phase"] + "…", state_val["progress_pct"]
                )
            ),  # progress_label
            gr.update(value=state_val["log_text"]),  # log_panel
            gr.update(visible=False),  # summary_section
            gr.update(value="Rendered"),  # view_mode
            gr.update(value=""),  # final_summary_md
            gr.update(value=""),  # final_summary_source
            gr.update(visible=False),  # copy_btn
            gr.update(value=None, visible=False),  # download_btn
            gr.update(interactive=False),  # run_btn
            state_val,
            gr.update(visible=stop_visible),  # stop_btn
        )

    def _announce_to_panel(call: Callable[[], None]) -> None:
        """Run an announcer helper with stdout redirected into the UI
        log panel AND the terminal at once.

        Announcer helpers (announce_start / announce_done /
        announce_unload / announce_unload_result) only ``print()`` to
        stdout. Inside a step, ``_stream_step`` already redirects
        stdout into the polled buffer that feeds state_val['log_text'].
        But announcer calls made by the main generator thread — start,
        done, and the finally block — run outside that capture, so the
        UI log panel wouldn't see them without explicit teeing. This
        helper is that tee: capture into a StringIO, append to
        log_text, and also forward to the real stdout so docker logs
        still get the line.
        """
        buf = io.StringIO()
        tee = _Tee(buf, sys.stdout)
        with contextlib.redirect_stdout(cast(TextIO, tee)):
            call()
        state_val["log_text"] += buf.getvalue()

    # ── Pre-flight: no transcript ────────────────────────────────────────
    if not state_val.get("canonical_md"):
        yield (
            gr.update(visible=True),
            gr.update(value=_progress_value("Cannot run", 0)),
            gr.update(value="❌ No transcript ingested. Upload a file first."),
            gr.update(visible=False),
            gr.update(value="Rendered"),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=True),
            state_val,
            gr.update(visible=False),
        )
        return

    # ── Pre-flight: bad models / unreachable host ────────────────────────
    ok, msg = preflight_check(ollama_host, editor_model, extractor_model)
    if not ok:
        yield (
            gr.update(visible=True),
            gr.update(value=_progress_value("Pre-flight failed", 0)),
            gr.update(value=f"❌ {msg}"),
            gr.update(visible=False),
            gr.update(value="Rendered"),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=True),
            state_val,
            gr.update(visible=False),
        )
        return

    tempdir = Path(state_val["tempdir_path"])
    canonical_md_path = Path(state_val["canonical_md"])

    state_val["models_used"].update({editor_model, extractor_model})
    _ALL_MODELS_EVER_LOADED.add((ollama_host, editor_model))
    _ALL_MODELS_EVER_LOADED.add((ollama_host, extractor_model))
    state_val["run_in_progress"] = True
    state_val["log_text"] = ""
    state_val["progress_pct"] = 0
    state_val["progress_phase"] = ""
    # M11: clear cached PDF path — a new run must regenerate on first PDF click.
    state_val["final_summary_pdf_path"] = None

    # Start banner — same shape as the MCP path so `docker compose
    # logs` tells the same story regardless of how the run was
    # invoked. Uses the uploaded stem when available (matches what
    # the user sees in the UI) and falls back to the canonical-md
    # filename otherwise.
    src_label = state_val.get("uploaded_stem") or canonical_md_path.name
    _announce_to_panel(lambda: announce_start("Gradio UI run", src_label))

    # Track whether the success branch already did its own unload
    # announcements (via _announce_to_panel, so they reach the UI log
    # panel). The finally block below checks this so we don't
    # double-unload on success, but still do a clean announcement on
    # exception or GeneratorExit (Stop button).
    unloaded_in_success_path = False

    try:
        # ── Step 1/4 in the UI's counting (step 2 in the pipeline):
        #    clean_transcript (editor_model, streamed) ────────────────────
        state_val["progress_pct"] = 0
        state_val["progress_phase"] = "Step 1/4 · Cleaning transcript"
        yield _running_tuple()
        announce(1, 4, "Cleaning transcript", editor_model)

        for _ in _stream_step(
            state_val,
            "_cleaned_path",
            lambda: clean_transcript(
                canonical_md_path,
                tempdir / "cleaned",
                editor_model,
                ollama_host,
            ),
        ):
            yield _running_tuple()
        cleaned_path = state_val["_cleaned_path"]

        # S3 fix: rename step 2's output to a stable stem before chaining.
        stable_cleaned_path = cleaned_path.parent / f"{STABLE_STEM}_cleaned.md"
        if cleaned_path != stable_cleaned_path:
            cleaned_path.rename(stable_cleaned_path)
            cleaned_path = stable_cleaned_path

        # ── Step 2/4: apply_speaker_mapping (pure, no model, instant) ────
        state_val["progress_pct"] = 25
        state_val["progress_phase"] = "Step 2/4 · Applying speaker names"
        yield _running_tuple()
        announce(2, 4, "Applying speaker names")

        cleaned_text = cleaned_path.read_text(encoding="utf-8")
        named_text = apply_speaker_mapping(cleaned_text, speaker_map or {})
        named_dir = tempdir / "named"
        named_dir.mkdir(parents=True, exist_ok=True)
        named_path = named_dir / f"{STABLE_STEM}_named.md"
        named_path.write_text(named_text, encoding="utf-8")

        # ── Step 3/4: extract_information (extractor_model, streamed) ────
        state_val["progress_pct"] = 50
        state_val["progress_phase"] = "Step 3/4 · Extracting information"
        yield _running_tuple()
        announce(3, 4, "Extracting intelligence", extractor_model)

        for _ in _stream_step(
            state_val,
            "_extracted_path",
            lambda: extract_information(
                named_path,
                tempdir / "extracted",
                extractor_model,
                ollama_host,
            ),
        ):
            yield _running_tuple()
        extracted_path = state_val["_extracted_path"]

        # ── Step 4/4: format_summary (extractor_model, streamed) ─────────
        state_val["progress_pct"] = 75
        state_val["progress_phase"] = "Step 4/4 · Formatting summary"
        yield _running_tuple()
        announce(4, 4, "Formatting final summary", extractor_model)

        for _ in _stream_step(
            state_val,
            "_final_path",
            lambda: format_summary(
                extracted_path,
                tempdir / "final",
                extractor_model,
                ollama_host,
            ),
        ):
            yield _running_tuple()
        final_path = state_val["_final_path"]

        stem = state_val.get("uploaded_stem") or final_path.stem
        download_dir = tempdir / "download"
        download_dir.mkdir(parents=True, exist_ok=True)
        download_path = download_dir / f"{stem}_summary.md"
        shutil.copy(final_path, download_path)
        state_val["final_summary_path"] = str(download_path)

        final_content = download_path.read_text(encoding="utf-8")

        # ── Terminal success ─────────────────────────────────────────────
        # No visibility toggling for final_summary_md / final_summary_source
        # any more — CSS controls which is displayed, driven by the
        # ``view-raw`` class on #summary-container. The md.change handler
        # (JS) fires when final_summary_md's value changes here, which
        # resets the container to rendered mode and re-clicks the radio
        # into the "Rendered" state.
        state_val["progress_pct"] = 100
        state_val["progress_phase"] = "Done"
        # Success banner — identical line to the MCP path's
        # announce_done(...), just a different destination clause. The
        # _announce_to_panel wrapper tees it into state_val['log_text']
        # so the browser log panel sees it too, not just the terminal.
        _announce_to_panel(
            lambda: announce_done(len(final_content), "Rendered in Web UI")
        )

        # Unload models IN-LINE (not in the finally block) on the
        # success path. Two reasons:
        #   1. The browser log panel only sees state_val["log_text"]
        #      writes that happen BEFORE the final yield. Anything in
        #      the finally block (which runs after the generator is
        #      exhausted) hits the terminal only, never the UI panel.
        #   2. The finally block remains as a safety net for the
        #      exception and GeneratorExit (Stop button) paths — a
        #      local flag below makes sure we don't double-unload.
        session_models = list(state_val.get("models_used", ()))
        _announce_to_panel(lambda: announce_unload(ollama_host, session_models))
        for m in session_models:
            try:
                unload_model(ollama_host, m)
                _announce_to_panel(lambda m=m: announce_unload_result(m, ok=True))
            except Exception as e:
                _announce_to_panel(
                    lambda m=m, e=e: announce_unload_result(
                        m, ok=False, error=str(e)
                    )
                )
        unloaded_in_success_path = True

        yield (
            gr.update(visible=True),  # console_group
            gr.update(value=_progress_value("✅ Done", 100)),  # progress_label
            gr.update(value=state_val["log_text"]),  # log_panel
            gr.update(visible=True),  # summary_section
            gr.update(value="Rendered"),  # view_mode
            gr.update(value=final_content),  # final_summary_md
            gr.update(value=final_content),  # final_summary_source
            gr.update(visible=True),  # copy_btn
            gr.update(value=str(download_path), visible=True),  # download_btn
            gr.update(interactive=True),  # run_btn
            state_val,
            gr.update(visible=False),  # stop_btn
        )

    except (Exception, SystemExit) as e:
        err_type = type(e).__name__
        err_msg = str(e) or err_type
        state_val["log_text"] += f"\n❌ {err_type}: {err_msg}\n"
        pct = state_val.get("progress_pct", 0)
        phase = state_val.get("progress_phase", "Pipeline")
        header = f"❌ Failed at {phase}" if phase else "❌ Failed"
        yield (
            gr.update(visible=True),
            gr.update(value=_progress_value(header, pct)),
            gr.update(value=state_val["log_text"]),
            gr.update(visible=False),
            gr.update(value="Rendered"),
            gr.update(value=""),
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=True),
            state_val,
            gr.update(visible=False),
        )

    finally:
        state_val["run_in_progress"] = False
        # VRAM cleanup path for exception + GeneratorExit (Stop button)
        # only — the success path already unloaded in-line (see above)
        # so its announcements reach the UI log panel, not just the
        # terminal. Python's finally always runs; the flag skips the
        # body cleanly on success.
        if not unloaded_in_success_path:
            session_models = list(state_val.get("models_used", ()))
            announce_unload(ollama_host, session_models)
            for m in session_models:
                try:
                    unload_model(ollama_host, m)
                    announce_unload_result(m, ok=True)
                except Exception as e:
                    announce_unload_result(m, ok=False, error=str(e))


# ─── Stateless Gradio API endpoint (M12.2) ───────────────────────────────


def markdown_to_pdf_endpoint(markdown: str) -> str:
    """Convert markdown text to a PDF file (M12.2 — stateless).

    Registered via ``gr.api(..., api_visibility="private")`` so it's
    reachable over HTTP but NOT listed in the MCP tool schema.
    Callable via ``gradio_client.Client`` or a direct
    ``POST /gradio_api/call/markdown_to_pdf``.

    Stateless: one fresh tempdir per call, no session binding, no
    cache. Callers that want session-aware PDF behaviour should use
    the Web UI's PDF radio path instead.

    Returns a filesystem path string; Gradio transparently turns that
    into a served file URL for the HTTP caller. Bytes-over-HTTP would
    be non-idiomatic in Gradio and would force callers into a custom
    response-handling path; the path-return convention is what
    ``gradio_client`` and the `/gradio_api/call/` shape both expect.

    Args:
        markdown: Markdown text body. Must not be empty / whitespace.

    Returns:
        Absolute path to the generated PDF file (inside a per-call
        tempdir under ``tempfile.gettempdir()``).

    Raises:
        ValueError: markdown is empty / whitespace only.
        OSError: write failure propagated from ``md_to_pdf``.
    """
    if not markdown or not markdown.strip():
        raise ValueError("Markdown source is empty.")
    tempdir = Path(tempfile.mkdtemp(prefix="mdpdf_api_"))
    out_path = tempdir / "summary.pdf"
    md_to_pdf(markdown, out_path)
    return str(out_path)


# ─── MCP-exposed tool ──────────────────────────────────────────────────

def on_view_mode_pdf(
    mode: str,
    state_val: dict[str, Any],
) -> Iterator[tuple[Any, Any, dict[str, Any]]]:
    """Handle view-mode radio changes that affect PDF display and the
    download-button target.

    Stacked alongside the JS-only ``view_mode.change`` listener
    (``TOGGLE_VIEW_MODE_JS``) that flips CSS classes. This handler runs
    in parallel, in Python, and owns two things:

      * The iframe HTML in ``#final-summary-pdf``.
      * The download button's target file + label + interactive state.

    Generator semantics — yields once for non-PDF modes / cached-PDF
    clicks, twice for first-PDF-click (placeholder + iframe).

    Yield tuple order matches the ``outputs=`` list on the wiring site:
      0. final_summary_pdf (HTML)
      1. download_btn (DownloadButton)
      2. session_state (State)
    """
    md_path = state_val.get("final_summary_path")

    # ── Rendered / Raw modes ────────────────────────────────────────
    # PDF iframe stays whatever it was (no update). Download button
    # points at the .md. If there's no summary yet, button is
    # disabled.
    if mode != "PDF":
        yield (
            gr.update(),  # iframe unchanged
            gr.update(
                value=md_path if md_path else None,
                label="Download",
                interactive=bool(md_path),
            ),
            state_val,
        )
        return

    # ── PDF mode ────────────────────────────────────────────────────
    if not md_path:
        # No summary run yet. Clear iframe, disable button.
        yield (
            gr.update(value=""),
            gr.update(interactive=False),
            state_val,
        )
        return

    pdf_path = state_val.get("final_summary_pdf_path")

    # Cached — serve iframe immediately, enable download as PDF.
    if pdf_path and Path(pdf_path).exists():
        # Gradio 6's static-file endpoint is `/gradio_api/file=<path>`
        # (the plain `/file=` prefix from Gradio 4.x is no longer a
        # reliable URL in 6.x). The path must also be whitelisted via
        # `allowed_paths=` in demo.launch() — we pass
        # `tempfile.gettempdir()` there so every per-session tempdir
        # rooted under it (e.g. /tmp/meeting_summarizer_XXX) is
        # reachable through this URL.
        iframe_html = (
            f'<iframe src="/gradio_api/file={pdf_path}" '
            f'style="width: 100%; height: {SUMMARY_HEIGHT}px; border: 0;" '
            f'title="Meeting summary PDF"></iframe>'
        )
        yield (
            gr.update(value=iframe_html),
            gr.update(
                value=pdf_path,
                label="Download PDF",
                interactive=True,
            ),
            state_val,
        )
        return

    # ── First click — placeholder, generate, then iframe ────────────
    # A2 placeholder: user sees "Generating PDF…" within ~50ms of the
    # click instead of a blank iframe. Option 2: download button is
    # DISABLED during the 1–3s generation window to avoid ambiguous
    # "is this MD or PDF right now?" state.
    placeholder_html = (
        '<div style="display: flex; align-items: center; '
        f'justify-content: center; height: {SUMMARY_HEIGHT}px; '
        'opacity: 0.6; font-style: italic;">Generating PDF…</div>'
    )
    yield (
        gr.update(value=placeholder_html),
        gr.update(interactive=False),
        state_val,
    )

    src_md = Path(md_path)
    target_pdf = src_md.with_suffix(".pdf")
    try:
        md_to_pdf(src_md, target_pdf)
    except Exception as e:
        # Conversion failed. Show red error banner in the iframe slot,
        # and fall the download button back to the .md (which is still
        # valid and available).
        err_html = (
            '<div style="display: flex; align-items: center; '
            f'justify-content: center; height: {SUMMARY_HEIGHT}px; '
            f'color: #ef4444; padding: 0 16px; text-align: center;">'
            f'PDF generation failed: {e}</div>'
        )
        yield (
            gr.update(value=err_html),
            gr.update(
                value=md_path,
                label="Download",
                interactive=True,
            ),
            state_val,
        )
        return

    state_val["final_summary_pdf_path"] = str(target_pdf)
    # Same URL scheme as the cached branch above — see comment there.
    iframe_html = (
        f'<iframe src="/gradio_api/file={target_pdf}" '
        f'style="width: 100%; height: {SUMMARY_HEIGHT}px; border: 0;" '
        f'title="Meeting summary PDF"></iframe>'
    )
    yield (
        gr.update(value=iframe_html),
        gr.update(
            value=str(target_pdf),
            label="Download PDF",
            interactive=True,
        ),
        state_val,
    )


def _materialize_input(src: str, dest_dir: Path) -> Path:
    """Resolve the ``file`` argument to a local file inside ``dest_dir``.

    Accepts three formats so MCP clients aren't forced to have
    filesystem access to the server:

      * ``data:<mime>;base64,<payload>`` — decoded and written to
        ``dest_dir``. Extension is ``.rtf`` if ``rtf`` appears in the
        mimetype, else ``.md`` (the two formats step 1 accepts).
      * ``http://...`` / ``https://...`` — streamed to ``dest_dir`` via
        httpx, preserving the URL's basename. Step 1 will reject if the
        resulting filename doesn't end in ``.md`` or ``.rtf``.
      * anything else — interpreted as a path on the SERVER's
        filesystem (NOT the MCP client's). Must exist.

    Raises:
        ValueError: for unknown prefixes, missing local files,
            malformed data URIs, or failed downloads.
    """
    src = (src or "").strip()
    if not src:
        raise ValueError("Empty `file` argument")

    if src.startswith("data:"):
        # data:<mime>;base64,<payload>
        try:
            header, payload = src.split(",", 1)
        except ValueError as e:
            raise ValueError("Malformed data URI (missing comma)") from e
        if ";base64" not in header:
            raise ValueError(
                "Only base64-encoded data URIs are supported "
                "(header must contain ';base64')"
            )
        mime = header[5:].split(";", 1)[0] or "text/plain"
        ext = ".rtf" if "rtf" in mime.lower() else ".md"
        try:
            data = base64.b64decode(payload, validate=True)
        except Exception as e:
            raise ValueError(f"Invalid base64 payload: {e}") from e
        out = dest_dir / f"upload{ext}"
        out.write_bytes(data)
        return out

    if src.startswith(("http://", "https://")):
        # Use the URL's path segment as the filename so step 1 can
        # detect the format via extension.
        url_path = src.split("?", 1)[0].rstrip("/")
        name = url_path.rsplit("/", 1)[-1] or "download"
        out = dest_dir / name
        print(f"Downloading {src} → {out.name}...")
        try:
            with httpx.stream(
                "GET", src, follow_redirects=True, timeout=30.0
            ) as r:
                r.raise_for_status()
                with open(out, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
        except httpx.HTTPError as e:
            raise ValueError(f"Download failed: {e}") from e
        return out

    # Otherwise: a path on the server's filesystem.
    #
    # Defensive pre-filter: if the caller pasted raw transcript text
    # into `file` by mistake (common MCP-client footgun), the string
    # will contain newlines and/or be enormous. Reject those early
    # with a clean ValueError that points them at `content=`, instead
    # of letting os.stat() throw ENAMETOOLONG / ENAMETOOLONG-like
    # OSErrors that surface as ugly stack traces to the MCP client.
    if "\n" in src or len(src) > 4096:
        preview = src[:80].replace("\n", " ") + ("…" if len(src) > 80 else "")
        raise ValueError(
            f"`file` argument doesn't look like a path, URL, or data URI "
            f"(got {len(src)} chars starting with {preview!r}). If you "
            f"meant to pass raw transcript text, use the `content` "
            f"parameter instead of `file`."
        )
    p = Path(src)
    try:
        exists = p.exists()
    except OSError as e:
        # Path-like string the OS can't even stat (too long, bad
        # chars, etc.). Treat as "not a valid path" and point the
        # caller at the right parameter.
        preview = src[:80] + ("…" if len(src) > 80 else "")
        raise ValueError(
            f"`file` argument {preview!r} is not a valid filesystem path "
            f"({type(e).__name__}: {e}). If you meant to pass raw "
            f"transcript text, use the `content` parameter instead."
        ) from e
    if not exists:
        preview = src[:80] + ("…" if len(src) > 80 else "")
        raise ValueError(
            f"Unrecognised `file` argument {preview!r} — expected a "
            f"'data:...', 'http(s)://...', or an existing local path on "
            f"the server's filesystem."
        )
    return p


def summarize_transcript(
    file: str | None = None,
    content: str | None = None,
    editor_model: str = DEFAULT_EDITOR_MODEL,
    extractor_model: str = DEFAULT_EXTRACTOR_MODEL,
    ollama_host: str | None = None,
    speaker_map: dict[str, str] | None = None,
) -> str:
    """Summarize a meeting transcript into polished meeting minutes.

    This is the MCP-exposed endpoint. Called by LLM tool-calling clients
    (Claude Desktop, Open WebUI, Cursor, etc.) via the Gradio MCP server
    at ``/gradio_api/mcp/``. The UI uses a separate generator pathway
    (``run_pipeline_generator``) that yields progress updates into
    Gradio components; this function is the headless equivalent that
    runs the full pipeline to completion and returns the final summary
    as a string.

    No per-step progress notifications are sent to MCP clients — they
    just wait 3–5 min for the return value. Terminal observability is
    preserved via the ``announce()`` step banners and the underlying
    pipeline ``print()`` output.

    **Important for agent callers:** action-item attribution quality
    depends on the transcript having real speaker names pre-assigned.
    When called without ``speaker_map``, any generic "Speaker N" tags
    are left as-is, which can lead to mis-attributed action items. For
    best results, either pre-name speakers in the transcript source,
    or pass ``speaker_map={"Speaker 1": "Alice", "Speaker 2": "Bob"}``.

    Args:
        file: Transcript source by reference. Accepts three formats so
            the MCP client doesn't need filesystem access to the
            server:

              * ``data:<mime>;base64,<payload>`` — base64-encoded file
                content inlined in the request. Use ``application/rtf``
                mimetype for Moonshine RTF exports; anything else is
                treated as markdown. Good for small transcripts.
              * ``http://...`` or ``https://...`` — public URL the
                server can fetch. Good for cross-machine tool calls
                (e.g. MCP client on a laptop hits a Gradio server on
                a LAN host; serve the transcript over
                ``python -m http.server`` or an equivalent).
              * anything else — an absolute path on the SERVER's
                filesystem (NOT the MCP client's). Must exist.

            Resulting filename must end in ``.rtf`` (Moonshine export)
            or ``.md`` (local transcriber output); step 1 rejects
            everything else.

            Mutually exclusive with ``content``. Exactly one must be
            provided.
        content: Transcript source by value. The raw string body of
            the transcript, already text-extracted. Written to a
            per-call tempfile with a ``.rtf`` extension if the content
            starts with ``{\rtf`` (for RTF source bodies), otherwise
            ``.md``. Then handed off to step 1 just like a real file
            upload would be.

            Primary use case: MCP clients like Open WebUI that extract
            text from chat-uploaded files into the LLM's context but
            do NOT expose a URL or path to tools. The calling LLM can
            paste the extracted text here verbatim (see issue #12228
            upstream; URL-based uploads are blocked pending that fix).

            Quality caveat: content passed this way has already been
            through the calling client's text extractor. For Moonshine
            RTFs extracted by Open WebUI, speaker labels (``**Speaker
            1:**``) may or may not survive the extractor depending on
            its configuration. For ``.md`` inputs from our local
            transcriber, round-tripping through an extractor is
            lossless. When speaker labels are lost, attribution in the
            final Action Items table degrades to ``(unnamed)`` — pass
            ``speaker_map`` if you can infer names from context.

            Mutually exclusive with ``file``. Exactly one must be
            provided.
        editor_model: Ollama model tag for the cleanup step (step 2).
            Defaults to ``gemma4:26b``.
        extractor_model: Ollama model tag for the extraction (step 4)
            and final-formatting (step 5) steps. Defaults to
            ``gemma4:26b``.
        ollama_host: Override for the server's ``OLLAMA_HOST`` env
            var. Omit to use the server default.
        speaker_map: Optional mapping of generic "Speaker N" tags to
            real names. Ignored if empty or None.

    Returns:
        The final meeting summary as a markdown string, including a
        Participants list, Executive Summary, Key Discussion Points,
        and Action Items table.

    Raises:
        ValueError: if neither or both of ``file`` and ``content`` are
            provided, if ``ollama_host`` is missing and no env default
            is configured, if the file cannot be located or ingested
            (unknown format, unreadable), or if Ollama is unreachable
            / either model is not pulled on the host. All loaded
            models are unloaded before the exception propagates.
        RuntimeError: if a pipeline step (cleanup, extraction,
            formatting) fails. All loaded models are unloaded before
            the exception propagates.
    """
    # Require exactly one of the two input sources. Both-or-neither
    # both fail with the same error so MCP clients get a consistent
    # "pick one" signal.
    if (file is None) == (content is None):
        raise ValueError(
            "Exactly one of `file` or `content` must be provided. "
            "Pass `file` as a URL / data URI / server path, OR pass "
            "`content` as the raw transcript text body."
        )

    # Resolve host. Explicit arg > env > fail.
    host = (ollama_host or DEFAULT_OLLAMA_HOST or "").strip()
    if not host:
        raise ValueError(
            "ollama_host not provided and OLLAMA_HOST env var not set"
        )

    # Pre-flight: reachability + both models pulled. Matches the UI's
    # Run-click gate so the two entry points fail for the same reasons.
    ok, msg = preflight_check(host, editor_model, extractor_model)
    if not ok:
        raise ValueError(msg)

    # Fresh tempdir per call — no session state to hang onto.
    tempdir = Path(tempfile.mkdtemp(prefix="meeting_summarizer_mcp_"))
    models_used = {editor_model, extractor_model}
    # Track so SIGTERM / atexit can unload even if this call is
    # interrupted mid-flight (matches UI behaviour).
    for m in models_used:
        _ALL_MODELS_EVER_LOADED.add((host, m))

    # Resolve the input source to a local file in tempdir. Two paths:
    #   * `content` — write the raw string to a .md (or .rtf if it
    #     begins with '{\rtf'), then let step 1 ingest it normally.
    #     Step 1's .md branch passes already-canonical markdown through
    #     unchanged, so no new pipeline code is needed here.
    #   * `file` — hand off to ``_materialize_input`` which handles
    #     data URIs, http(s) URLs, or server filesystem paths.
    raw_dir = tempdir / "raw_files"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if content is not None:
        # Very loose RTF sniff — a real RTF always starts with the
        # literal bytes `{\rtf`. Anything else is treated as markdown
        # / plain text, which step 1's .md handler can pass through
        # when it's in canonical `**Speaker N:**` form.
        ext = ".rtf" if content.lstrip().startswith(r"{\rtf") else ".md"
        input_path = raw_dir / f"upload{ext}"
        try:
            input_path.write_text(content, encoding="utf-8")
        except OSError as e:
            shutil.rmtree(tempdir, ignore_errors=True)
            raise ValueError(f"Failed to stage `content` to tempdir: {e}") from e
    else:
        try:
            input_path = _materialize_input(file, raw_dir)
        except ValueError:
            # Nothing loaded yet — just clean the tempdir and re-raise.
            shutil.rmtree(tempdir, ignore_errors=True)
            raise

    # Terminal banner so CLI operators watching `uv run app.py` see a
    # clear start / end for each MCP call. Mirrors main.py's
    # "Starting Pipeline for: ..." line. Shape is shared with the UI
    # path via pipeline.announce_start / announce_done /
    # announce_unload(_result) so both entry points produce identical
    # log output.
    announce_start("MCP summarize_transcript", input_path.name)

    try:
        # ── Step 1/5: ingest ──────────────────────────────────────────────────
        announce(1, 5, "Ingesting transcript to Markdown")
        # raw_dir already exists from _materialize_input; reuse it so
        # the materialized input and step1's normalized output sit
        # alongside each other.
        try:
            _, canonical_md = convert(input_path, raw_dir)
        except SystemExit as e:
            # step1 signals format rejection via sys.exit — translate
            # to a proper exception type for MCP clients.
            raise ValueError(
                f"Could not ingest {input_path.name}: "
                f"step 1 rejected the file ({e})"
            ) from e

        # ── Step 2/5: cleanup ─────────────────────────────────────────────────
        announce(2, 5, "Cleaning transcript", editor_model)
        cleaned_path = clean_transcript(
            canonical_md, tempdir / "cleaned", editor_model, host
        )
        # Stable stem keeps downstream filenames predictable when the
        # source stem contains ".named" (S3 fix; applies equally here).
        stable_cleaned = cleaned_path.parent / f"{STABLE_STEM}_cleaned.md"
        if cleaned_path != stable_cleaned:
            cleaned_path.rename(stable_cleaned)
            cleaned_path = stable_cleaned

        # ── Step 3/5: speaker mapping (pure, no model) ──────────────────────
        announce(3, 5, "Applying speaker names")
        cleaned_text = cleaned_path.read_text(encoding="utf-8")
        named_text = apply_speaker_mapping(cleaned_text, speaker_map or {})
        named_dir = tempdir / "named"
        named_dir.mkdir(parents=True, exist_ok=True)
        named_path = named_dir / f"{STABLE_STEM}_named.md"
        named_path.write_text(named_text, encoding="utf-8")

        # ── Step 4/5: extraction ────────────────────────────────────────────────
        announce(4, 5, "Extracting intelligence", extractor_model)
        extracted_path = extract_information(
            named_path, tempdir / "extracted", extractor_model, host
        )

        # ── Step 5/5: format ───────────────────────────────────────────────────
        announce(5, 5, "Formatting final summary", extractor_model)
        final_path = format_summary(
            extracted_path, tempdir / "final", extractor_model, host
        )

        final_text = final_path.read_text(encoding="utf-8")

        # Closing banner — matches announce_start's separator; the UI
        # path emits the same line via announce_done below.
        announce_done(len(final_text), "Returning to MCP client")
        return final_text

    except ValueError:
        # Pre-flight / bad-file errors already carry the right type —
        # pass through.
        raise
    except (Exception, SystemExit) as e:
        # Everything else (step failures, Ollama mid-call errors) gets
        # wrapped as RuntimeError so MCP clients get a uniform
        # "pipeline blew up" signal.
        raise RuntimeError(
            f"Pipeline step failed: {type(e).__name__}: {e}"
        ) from e
    finally:
        # Eject models on every exit path — success, raise, or cancel.
        # unload_model swallows transport errors so we re-do the HTTP
        # call here behind try/except to know whether to report
        # success or a note. Shared announcer helpers keep the
        # terminal output identical to the UI path.
        announce_unload(host, models_used)
        for m in models_used:
            try:
                unload_model(host, m)
                announce_unload_result(m, ok=True)
            except Exception as e:
                announce_unload_result(m, ok=False, error=str(e))
        shutil.rmtree(tempdir, ignore_errors=True)


# ─── UI construction ──────────────────────────────────────────────────────


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Meeting Transcription Summarizer (Local)") as demo:
        session_state = gr.State(
            value=init_session_state(),
            delete_callback=cleanup_session,
        )
        all_speakers_state = gr.State([])
        speaker_map_state = gr.State({})

        # ── Sidebar ──────────────────────────────────────────────────────
        with gr.Sidebar(open=False):
            gr.Markdown("### Settings")

            gr.Markdown("**Ollama host**")
            ollama_host = gr.Textbox(
                value=DEFAULT_OLLAMA_HOST,
                placeholder="http://<host>:<port>",
                show_label=False,
                container=False,
            )
            with gr.Row(elem_id="sidebar-host-row"):
                # FontAwesome arrow-rotate-right SVG (see assets/refresh.svg).
                # value="" + icon= renders an icon-only button; CSS in
                # CUSTOM_CSS keeps it compact and sizes the <img>.
                test_btn = gr.Button(
                    value="",
                    icon=REFRESH_ICON_PATH,
                    scale=0,
                    min_width=40,
                    variant="secondary",
                    elem_id="test-btn",
                )
                connection_indicator = gr.HTML(
                    value=_connection_indicator_html(DEFAULT_OLLAMA_HOST),
                    elem_id="conn-indicator",
                )

            editor_model = gr.Textbox(
                label="Editor model (step 2)",
                value=DEFAULT_EDITOR_MODEL,
                info="Used for cleanup.",
            )
            editor_status = gr.Markdown("", elem_id="editor-status")

            extractor_model = gr.Textbox(
                label="Extractor model (steps 4 & 5)",
                value=DEFAULT_EXTRACTOR_MODEL,
                info="Used for information extraction + final formatting.",
            )
            extractor_status = gr.Markdown("", elem_id="extractor-status")

        # ── Main column ──────────────────────────────────────────────────
        with gr.Column():
            banner = gr.Markdown("", visible=False, elem_id="ollama-banner")

            gr.Markdown("# Meeting Transcription Summarizer (Local)")
            gr.Markdown(
                "Upload a meeting transcript as **`.rtf`** "
                "(from [moonshine-notetaker](https://note-taker.moonshine.ai/)) "
                "or **`.md`** (from the zz's local transcriber). "
                "Max 10 MB; practical ceiling is ≈2.5h of speech "
                "(the LLM's context window)."
            )

            with gr.Accordion("⚠️ IMPORTANT ⚠️", open=True):
                gr.Markdown(
                    "Each tab runs independently — multiple tabs = multiple "
                    "queue slots. When you close the tab, the models your "
                    "session loaded and its temp files are cleaned up "
                    "automatically (within about an hour). "
                    "ENGLISH GIVES BEST RESULTS"
                )

            # Label carries the file-type + size constraint so it's
            # visible on the dropzone itself, not just in the prose
            # above. file_types also filters the native file picker,
            # but drag-and-drop still accepts anything — step 1 will
            # reject unknown formats with a clear error_md message.
            upload = gr.File(
                label="Transcript  ·  .rtf or .md  ·  max 10 MB",
                file_types=[".rtf", ".md"],
                file_count="single",
                elem_id="transcript-upload",
            )

            error_md = gr.Markdown("", visible=False, elem_id="ingest-error")
            meta_md = gr.Markdown("", visible=False, elem_id="ingest-meta")

            # ── L-C split: Preview (left) + Speakers (right) ─────────────
            with gr.Row():
                with gr.Column():
                    preview_md = gr.Markdown(
                        "",
                        label="Preview",
                        height=PREVIEW_HEIGHT,
                        max_height=PREVIEW_HEIGHT,
                        visible=False,
                        container=True,
                        padding=True,
                        elem_id="transcript-preview",
                    )

                with gr.Column():

                    @gr.render(inputs=[all_speakers_state])
                    def render_speaker_form(
                        all_speakers: list[tuple[str, bool]],
                    ):
                        if not all_speakers:
                            return
                        gr.Markdown("### Speaker names")
                        for name, is_generic in all_speakers:
                            if is_generic:
                                tb = gr.Textbox(
                                    label=name,
                                    placeholder="Enter a real name, or leave blank",
                                    interactive=True,
                                )

                                def _make_updater(captured_tag: str):
                                    def _update(
                                        new_val: str,
                                        current_map: dict[str, str],
                                    ):
                                        new_map = dict(current_map or {})
                                        if new_val and new_val.strip():
                                            new_map[captured_tag] = new_val.strip()
                                        else:
                                            new_map.pop(captured_tag, None)
                                        return new_map

                                    return _update

                                tb.change(
                                    _make_updater(name),
                                    inputs=[tb, speaker_map_state],
                                    outputs=[speaker_map_state],
                                    api_visibility="private",
                                )
                            else:
                                gr.Textbox(
                                    label=f"{name} (already named)",
                                    value=name,
                                    interactive=False,
                                )

            # ── Run + Stop row ───────────────────────────────────────────
            with gr.Row():
                run_btn = gr.Button(
                    "Run",
                    variant="primary",
                    interactive=False,
                    elem_id="run-btn",
                )
                stop_btn = gr.Button(
                    "Stop",
                    variant="stop",
                    visible=False,
                    elem_id="stop-btn",
                )

            # ── Console panel: progress + log ────────────────────────────
            with gr.Group(visible=False) as console_group:
                progress_label = gr.Label(
                    value={},
                    show_heading=False,
                    show_label=False,
                    container=False,
                    elem_id="progress-label",
                )
                log_panel = gr.Textbox(
                    value="",
                    label="",
                    show_label=False,
                    lines=LOG_PANEL_LINES,
                    max_lines=LOG_PANEL_LINES,
                    autoscroll=True,
                    interactive=False,
                    elem_id="log-panel",
                    elem_classes=["log-panel"],
                    placeholder="",
                )

            # ── Results section ──────────────────────────────────────────
            # elem_id="summary-container" — CSS targets this plus a
            # runtime-toggled ``.view-raw`` class to flip display between
            # the two child components. Both children stay visible=True
            # at the Gradio level; CSS does the work.
            with gr.Column(
                variant="panel",
                visible=False,
                elem_id="summary-container",
            ) as summary_section:
                view_mode = gr.Radio(
                    choices=["Rendered", "Raw", "PDF"],
                    value="Rendered",
                    show_label=False,
                    container=False,
                    interactive=True,
                    elem_id="view-mode",
                )

                final_summary_md = gr.Markdown(
                    "",
                    label="Meeting summary",
                    height=SUMMARY_HEIGHT,
                    max_height=SUMMARY_HEIGHT,
                    container=True,
                    padding=True,
                    elem_id="final-summary",
                )

                final_summary_source = gr.Textbox(
                    value="",
                    elem_id="final-summary-source",
                    interactive=True,
                    show_label=False,
                    lines=SUMMARY_RAW_LINES,
                    max_lines=SUMMARY_RAW_LINES,
                )

                # M11: iframe slot for PDF view mode. Visible in the
                # Python/Gradio sense at all times; CSS
                # (.view-pdf on #summary-container) controls actual
                # display. Populated by on_view_mode_pdf on demand.
                final_summary_pdf = gr.HTML(
                    value="",
                    elem_id="final-summary-pdf",
                    visible=True,
                )

                with gr.Row():
                    copy_btn = gr.Button(
                        "Copy",
                        variant="secondary",
                        visible=False,
                        elem_id="copy-btn",
                    )
                    download_btn = gr.DownloadButton(
                        "Download",
                        visible=False,
                        elem_id="download-btn",
                    )

            # ── Footer ───────────────────────────────────────────────────
            # Sits above Gradio's own built-in footer (the small "Built
            # with Gradio" strip). Gradio 6 has no API to add items TO
            # the built-in footer, so we append our own attribution line
            # at the bottom of the main column instead. CSS in
            # CUSTOM_CSS keeps it muted, centre-aligned, and with a bit
            # of top margin so it doesn't hug the Download button.
            gr.Markdown(
                (
                    "[Saurabh Datta](https://github.com/dattasaurabh82) · "
                    "[zigzag.is](https://zigzag.is) · "
                    "[GitHub](https://github.com/dattazigzag/local-meeting-transcript-summerizer)"
                ),
                elem_id="app-footer",
            )

        # ── Event wiring ──────────────────────────────────────────────────

        demo.load(
            on_startup,
            inputs=[session_state],
            outputs=[banner, editor_status, extractor_status, connection_indicator],
            api_visibility="private",
        )
        demo.load(fn=None, inputs=None, outputs=None, js=FORCE_DARK_MODE_JS)

        ollama_host.change(
            on_host_change,
            inputs=[ollama_host, editor_model, extractor_model, session_state],
            outputs=[
                session_state,
                banner,
                editor_status,
                extractor_status,
                connection_indicator,
            ],
            api_visibility="private",
        )

        editor_model.change(
            _model_indicator,
            inputs=[ollama_host, editor_model],
            outputs=[editor_status],
            api_visibility="private",
        )
        extractor_model.change(
            _model_indicator,
            inputs=[ollama_host, extractor_model],
            outputs=[extractor_status],
            api_visibility="private",
        )

        test_btn.click(
            on_test_connection,
            inputs=[ollama_host],
            outputs=[connection_indicator, banner],
            api_visibility="private",
        )

        # Rendered/Raw/PDF toggle — two listeners stacked on the same
        # event. The JS listener flips CSS classes (instant, no Python
        # roundtrip). The Python listener (on_view_mode_pdf) handles
        # PDF generation on demand + updates the download button's
        # target. Gradio runs both in parallel.
        view_mode.change(
            fn=None,
            inputs=[view_mode],
            outputs=None,
            js=TOGGLE_VIEW_MODE_JS,
        )
        view_mode.change(
            on_view_mode_pdf,
            inputs=[view_mode, session_state],
            outputs=[final_summary_pdf, download_btn, session_state],
            api_visibility="private",
        )

        # When the rendered markdown's value changes (success yield sets
        # it to the final summary, reset yields clear it), force the
        # container back to rendered mode and the radio back to Rendered.
        # Handles the "both views visible after success" bug that came
        # from Gradio's radio .change() not firing on programmatic value
        # updates.
        final_summary_md.change(
            fn=None,
            inputs=None,
            outputs=None,
            js=RESET_VIEW_MODE_JS,
        )

        upload.change(
            on_file_upload,
            inputs=[upload, session_state],
            outputs=[
                preview_md,
                error_md,
                run_btn,
                all_speakers_state,
                speaker_map_state,
                session_state,
                meta_md,
                # 6-entry run-output reset:
                console_group,
                progress_label,
                log_panel,
                summary_section,
                copy_btn,
                download_btn,
            ],
            api_visibility="private",
        )

        run_event = run_btn.click(
            run_pipeline_generator,
            inputs=[
                session_state,
                editor_model,
                extractor_model,
                ollama_host,
                speaker_map_state,
            ],
            outputs=[
                console_group,
                progress_label,
                log_panel,
                summary_section,
                view_mode,
                final_summary_md,
                final_summary_source,
                copy_btn,
                download_btn,
                run_btn,
                session_state,
                stop_btn,
            ],
            api_visibility="private",
        )

        # 9-tuple match for on_stop (no longer needs view_mode /
        # final_summary_md / final_summary_source entries — the md.change
        # JS handler resets them when final_summary_md's value toggles
        # to "").
        stop_btn.click(
            on_stop,
            inputs=[session_state],
            outputs=[
                console_group,
                progress_label,
                log_panel,
                summary_section,
                copy_btn,
                download_btn,
                run_btn,
                session_state,
                stop_btn,
            ],
            cancels=[run_event],
            api_visibility="private",
        )

        copy_btn.click(
            None,
            inputs=None,
            outputs=None,
            js=COPY_SUMMARY_JS,
        )

        # ── MCP-exposed tool endpoint ─────────────────────────────────────────
        # ``gr.api`` registers a Gradio endpoint without binding it to a
        # UI component — pure logic in, pure data out. When the app is
        # launched with ``mcp_server=True`` this surfaces as the ONE tool
        # exposed to MCP clients at ``/gradio_api/mcp/``. All other event
        # listeners above carry ``api_visibility="private"`` so they
        # don't pollute the tool list. The four listeners with ``fn=None``
        # (the JS-only ones: dark-mode load, view-mode toggle,
        # final_summary_md.change, copy button) are auto-private per
        # Gradio 6 — see gr.Blocks docs: "If fn is None, api_visibility
        # will automatically be set to 'private'."
        gr.api(summarize_transcript, api_name="summarize_transcript")

        # M12.2: stateless markdown→PDF endpoint. HTTP-callable but
        # NOT in the MCP tool list (api_visibility="private"). Callers
        # who already have a markdown string (e.g. the output of a
        # prior summarize_transcript call) can pipe it through this
        # endpoint to get a PDF back, without a browser session.
        gr.api(
            markdown_to_pdf_endpoint,
            api_name="markdown_to_pdf",
            api_visibility="private",
        )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI flags for the Gradio + MCP server.

    All flags are optional — ``uv run app.py`` with no args preserves the
    pre-docker bare-metal behaviour (bind 0.0.0.0:7860, read OLLAMA_HOST
    from env, default both models to ``gemma4:26b``).

    The docker CMD injects ``--port 2070`` to pin the container to the
    ziggie 20xx port band. The ``--host`` default (0.0.0.0) is already
    what we want inside a container, so the CMD doesn't set it.

    Precedence for each value: CLI flag > env var > hardcoded fallback.
    """
    parser = argparse.ArgumentParser(
        prog="app.py",
        description=(
            "Gradio + MCP server for the Local Meeting Transcript "
            "Summarizer. Runs on bare metal (no flags → http://localhost"
            ":7860) or in a container (docker CMD injects --host/--port)."
        ),
    )

    # ── Server binding ───────────────────────────────────────────────
    # Defaults chosen so bare-metal `uv run app.py` preserves today's
    # behaviour. Env vars honour Gradio's own conventions
    # (GRADIO_SERVER_NAME / GRADIO_SERVER_PORT) so they also work if
    # exported without flags.
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("GRADIO_SERVER_NAME", SERVER_HOST),
        help=(
            "Gradio bind address. Default: $GRADIO_SERVER_NAME or "
            f"{SERVER_HOST!r}."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("GRADIO_SERVER_PORT", SERVER_PORT)),
        help=(
            "Gradio bind port. Default: $GRADIO_SERVER_PORT or "
            f"{SERVER_PORT}."
        ),
    )

    # ── Pipeline defaults ────────────────────────────────────────────
    # These override the module-level DEFAULT_* constants that both the
    # sidebar widgets AND summarize_transcript's signature defaults read
    # from. See main() for how the override is applied.
    #
    # Naming note: main.py's CLI calls its Ollama flag --host (its only
    # host-ish concept). Here --host is the Gradio bind address, so we
    # disambiguate the Ollama URL as --ollama-host. Slight divergence
    # between the two entry points but each one's flags are internally
    # consistent.
    parser.add_argument(
        "--ollama-host",
        type=str,
        default=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        help=(
            "Ollama server URL (used for connectivity probes, model "
            "listing, and as the default passed to MCP callers that omit "
            "the arg). Default: $OLLAMA_HOST. Hard-fails if empty."
        ),
    )
    parser.add_argument(
        "--editor-model",
        type=str,
        default=DEFAULT_EDITOR_MODEL,
        help=(
            "Default model for Cleanup (step 2). Shown in the sidebar; "
            "also the default for MCP calls that omit editor_model. "
            f"Default: {DEFAULT_EDITOR_MODEL!r}."
        ),
    )
    parser.add_argument(
        "--extractor-model",
        type=str,
        default=DEFAULT_EXTRACTOR_MODEL,
        help=(
            "Default model for Extraction (step 4) and Formatting (step "
            "5). Shown in the sidebar; also the default for MCP calls "
            f"that omit extractor_model. Default: {DEFAULT_EXTRACTOR_MODEL!r}."
        ),
    )

    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()

    # Apply CLI-resolved values by mutating the module-level DEFAULT_*
    # constants. Ugly but unavoidable: summarize_transcript's signature
    # defaults are frozen at function definition time (they read the
    # constants at import), so we must update the constants before the
    # MCP tool is registered via gr.api(). The sidebar widgets also read
    # these constants for their initial value=.
    #
    # Bare-metal `uv run app.py` with no flags leaves everything pinned
    # to the pre-mutation defaults (argparse defaults fall through to
    # the original constant values).
    global DEFAULT_OLLAMA_HOST, DEFAULT_EDITOR_MODEL, DEFAULT_EXTRACTOR_MODEL
    DEFAULT_OLLAMA_HOST = (args.ollama_host or "").strip()
    DEFAULT_EDITOR_MODEL = args.editor_model
    DEFAULT_EXTRACTOR_MODEL = args.extractor_model

    # Also rebind summarize_transcript's frozen defaults so MCP callers
    # that omit the args pick up the CLI-chosen models. __defaults__ is
    # a tuple (file, content, editor_model, extractor_model,
    # ollama_host, speaker_map) for the signature
    # (file=None, content=None, editor_model=..., extractor_model=...,
    #  ollama_host=None, speaker_map=None). We only rewrite the two
    # model slots; file/content/ollama_host/speaker_map stay None
    # (summarize_transcript reads DEFAULT_OLLAMA_HOST from the module
    # namespace at call-time, not sig-time).
    old_defaults = summarize_transcript.__defaults__
    if old_defaults is not None:
        # (file, content, editor_model, extractor_model, ollama_host,
        #  speaker_map) — six entries now that `file` and `content`
        # are both optional (None-default) input sources.
        summarize_transcript.__defaults__ = (
            old_defaults[0],  # file           — unchanged (None)
            old_defaults[1],  # content        — unchanged (None)
            DEFAULT_EDITOR_MODEL,
            DEFAULT_EXTRACTOR_MODEL,
            old_defaults[4],  # ollama_host    — unchanged (None)
            old_defaults[5],  # speaker_map    — unchanged (None)
        )

    if not DEFAULT_OLLAMA_HOST:
        print(
            "Error: OLLAMA_HOST is missing. Pass --ollama-host or set "
            "OLLAMA_HOST in .env / the environment.",
            file=sys.stderr,
        )
        sys.exit(1)

    _install_process_hooks()

    demo = build_demo()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        # Do not auto-open a browser tab. Matters for headless hosts
        # like ziggie where there IS no desktop, and is just noisy on
        # Mac dev. Users hit http://<host>:<port> manually.
        inbrowser=False,
        theme=gr.Theme.from_hub("Nymbo/Nymbo_Theme"),
        css=CUSTOM_CSS,
        max_file_size=UPLOAD_MAX_SIZE,
        mcp_server=True,
        # M11: whitelist the tempfile root so the PDF iframe's
        # `<iframe src="/gradio_api/file=/tmp/...">` URL resolves.
        # Without this, Gradio 6 returns {"detail": "Not Found"} for
        # arbitrary server paths (security hardening since 5.x). Our
        # per-session tempdirs all sit under tempfile.gettempdir() —
        # usually `/tmp` on Linux/macOS or `/var/folders/…` on some
        # macOS configs — so whitelisting that root covers every
        # session without leaking anything user-sensitive (the
        # tempdir is process-scoped and cleaned up on session end).
        # `DownloadButton` uses Gradio's internal file-token system,
        # not this URL route, so it worked before this fix — only
        # the iframe needed it.
        allowed_paths=[tempfile.gettempdir()],
    )


if __name__ == "__main__":
    main()
