#!/usr/bin/env python3
"""
Gradio front-end for the Local Meeting Transcript Summarizer.

Thin skin over ``main.py``'s pipeline. See ``contexts/gradio_app.md`` for the
full spec, decisions, and milestone breakdown.

Run:
    uv run app.py

Then open http://localhost:7860 in a browser.

Status: M6 complete · M6.5 in progress.
This pass (M6.5 round 3):
  * gr.Label replaces the hand-rolled HTML progress bar.
  * Streaming log capture via a background thread + stdout tee.
  * Preview + speaker form side-by-side (layout L-C).
  * Speaker form always rendered; pre-named speakers shown disabled.
  * Rendered/Raw toggle visibility now resets deterministically.
  * Sidebar host row split into [textbox] / [refresh + LED] for alignment.
  * LED is a CSS-coloured circle, not an emoji glyph.
"""

from __future__ import annotations

import atexit
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
from typing import Any, Callable, Iterator

import gradio as gr
import httpx
from dotenv import load_dotenv

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

# Fixed preview/summary heights. gr.Markdown(height=N, max_height=N) +
# container=True gives theme-bordered boxes that scroll internally past N
# pixels. Same height on summary + raw textbox so the Rendered/Raw toggle
# doesn't jump the page.
PREVIEW_HEIGHT = 400
SUMMARY_HEIGHT = 500
SUMMARY_RAW_LINES = 20

# Threaded-streaming polling interval for stdout capture (M6.5 L1 streaming
# upgrade). 0.3s trades per-step overhead (~300-500 extra UI updates per
# long Ollama call) against responsiveness (log lines appear within 300ms
# of the step's print). Lower values feel snappier but pile up more yields.
# 0.3 is comfortable for humans and modest for the browser.
LOG_POLL_INTERVAL = 0.3


# ─── Regex for speaker detection ─────────────────────────────────────────

# Matches "**Name:**" at the start of a line. Name is everything up to the
# first colon. Pipeline already guarantees this format from step1_convert.
_SPEAKER_LINE_RE = re.compile(r"^\*\*([^:]+):\*\*")
# "Speaker N" where N is one or more digits. Generic placeholder produced
# by upstream transcription tools. Non-matches = real names that the user
# (or prior transcription) already set.
_GENERIC_SPEAKER_RE = re.compile(r"^Speaker \d+$")


def detect_all_speakers(md_text: str) -> list[tuple[str, bool]]:
    """Return [(speaker_name, is_generic), ...] preserving first-seen order.

    ``is_generic`` is True when the name matches 'Speaker N'. The UI uses
    this to decide whether to render each textbox as editable (generic,
    user needs to fill in) or disabled (already named, just display).

    Previously the app only detected generic speakers via
    ``detect_generic_speakers()`` from the pipeline module; that hid the
    pre-named ones entirely. Showing them disabled gives the user a
    complete view of every speaker the pipeline will see, which helps
    when mapping voices in the preview to the form (M6.5 L-C fix).
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
# constructor. Three narrow scopes:
#
#   .log-panel textarea — monospace log panel font. Pulls var(--font-mono)
#   (Nymbo: JetBrains Mono) with explicit fallback stack.
#
#   #conn-indicator — centres the LED dot vertically inside its row cell.
#   The dot itself is a small coloured <div> emitted by
#   ``_connection_indicator_html`` and rendered via gr.HTML (not an emoji
#   — macOS renders 🟢 as a flat square in sans-serif contexts). Colour
#   comes from inline style so state transitions don't need a class swap.
#
#   #sidebar-host-row — tightens the spacing between the refresh button
#   and LED indicator in the sidebar.
#
# No progress-bar CSS is needed any more — gr.Label provides the bar +
# percentage + theme colour natively (M6.5 round 3 pivot). Deleted the
# .progress-* rules that supported the previous custom gr.HTML approach.
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
"""


# ─── Dark-mode forcing ────────────────────────────────────────────────────
FORCE_DARK_MODE_JS = """
() => {
    if (!document.body.classList.contains('dark')) {
        document.body.classList.add('dark');
    }
}
"""

# JS fired by the Copy button. Reads raw markdown source from the hidden
# gr.Textbox rather than the rendered markdown's innerText so the clipboard
# gets markdown syntax intact. (M6.5 S1)
COPY_SUMMARY_JS = """
() => {
    const host = document.getElementById('final-summary-source');
    if (!host) {
        console.warn('final-summary-source element not found');
        return;
    }
    const ta = host.querySelector('textarea');
    if (!ta) {
        console.warn('final-summary-source textarea not found');
        return;
    }
    navigator.clipboard.writeText(ta.value || '');
}
"""


# ─── Threaded stdout streaming helpers (M6.5 L1 upgrade) ─────────────────

class _StreamingBuffer:
    """Thread-safe text buffer used as stdout during a step run.

    Concurrent ``write`` calls from the step thread and ``getvalue`` calls
    from the main (generator) thread are serialised via a lock. CPython's
    GIL plus StringIO's underlying list-of-chunks is probably race-safe
    already, but explicit locking makes the intent clear and survives
    future CPython internal changes.
    """

    def __init__(self) -> None:
        self._buf = io.StringIO()
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            return self._buf.write(s)

    def flush(self) -> None:
        with self._lock:
            self._buf.flush()

    def getvalue(self) -> str:
        with self._lock:
            return self._buf.getvalue()


def _stream_step_into_state(
    state_val: dict[str, Any],
    result_key: str,
    fn: Callable[[], Any],
) -> Iterator[None]:
    """Run ``fn`` in a daemon thread with stdout captured; yield each time
    new output appears so the calling generator can refresh the log panel.

    Side effects on ``state_val``:
        * ``state_val['log_text']`` is appended to with each captured chunk
          (initial log preserved so previous steps' output stays visible).
        * ``state_val[result_key]`` is set to ``fn``'s return value on
          successful completion.

    Yields None; callers should re-emit their full Gradio output tuple
    after each yield (the log_text on state has been refreshed by then).

    Exceptions raised inside ``fn`` are captured and re-raised from this
    generator at the end of iteration, so the caller's try/except/finally
    sees them normally. ``SystemExit`` from the pipeline's sys.exit(1)
    paths is treated like any other exception (caught by the generator's
    own except clause in run_pipeline_generator).

    Known caveat: ``contextlib.redirect_stdout`` patches ``sys.stdout``
    globally — writes from *any* thread during the redirect window land
    in this buffer, not just the step thread. Single-user usage is
    unaffected. Multi-user concurrent runs would interleave captures;
    revisit if that becomes a real scenario (deferred from M7+).

    Generator cancellation (GeneratorExit from Gradio's ``cancels=``)
    interrupts the main thread's yield, not the step thread. The step
    thread keeps running in the background until its Ollama call returns,
    then exits normally. Models unload in the outer try/finally. This
    matches the spec F3 cancellation semantics.
    """
    buf = _StreamingBuffer()
    result_holder: list[Any] = [None]
    exception_holder: list[BaseException | None] = [None]

    def _target() -> None:
        try:
            with contextlib.redirect_stdout(buf):
                result_holder[0] = fn()
        except BaseException as e:
            exception_holder[0] = e

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()

    initial_log = state_val.get("log_text", "")
    last_captured_len = 0

    # Poll the buffer while the step runs. Only yield when there's new
    # content to avoid redundant UI updates.
    while thread.is_alive():
        time.sleep(LOG_POLL_INTERVAL)
        current = buf.getvalue()
        if len(current) > last_captured_len:
            state_val["log_text"] = initial_log + current
            yield
            last_captured_len = len(current)

    # Thread has finished. Small join timeout guards against races between
    # is_alive() returning False and the thread fully exiting.
    thread.join(timeout=1.0)

    # Final flush — catch anything written between the last poll and
    # thread termination.
    final = buf.getvalue()
    if len(final) != last_captured_len:
        state_val["log_text"] = initial_log + final
        yield

    if exception_holder[0] is not None:
        raise exception_holder[0]
    state_val[result_key] = result_holder[0]


# ─── Progress-bar helper (M6.5 round 3 — gr.Label pivot) ─────────────────

def _progress_value(phase: str, pct: int) -> dict[str, float]:
    """Build the value dict for gr.Label.

    gr.Label renders a ``dict[str, float]`` as a labelled horizontal bar:
    key = label text, value = fraction of full width. Percentage text
    ("50%") and theme-accent bar colour are rendered by gr.Label
    automatically, which is the whole reason for the pivot from the
    earlier custom gr.HTML approach — no custom CSS, matches the theme,
    and looks exactly like the example gr.Label screenshot.

    ``pct`` is clamped to [0, 100] and converted to 0.0–1.0 for gr.Label.
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
    """Factory for ``gr.State``. See contexts/gradio_app.md for full shape.

    ``log_text`` / ``progress_pct`` / ``progress_phase`` let on_stop
    reconstruct the cancellation view at the exact point Stop was hit.
    """
    return {
        "tempdir_path": None,
        "canonical_md": None,
        "uploaded_stem": None,
        "final_summary_path": None,
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
    """Per-session cleanup via ``gr.State.delete_callback`` (~60 min after
    tab close). See spec changelog (M6 fix-up) for why the arg is optional.
    """
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

    Previous attempt used 🟢 / 🔴 emoji inside gr.Markdown. On macOS browsers
    those glyphs render square in sans-serif contexts — a font-fallback
    artifact, not a Gradio issue. Switching to a plain coloured <div> with
    ``border-radius: 50%`` sidesteps the entire emoji-font path and
    guarantees a circle everywhere.
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
    """``demo.load`` handler."""
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
    """Compact refresh button handler."""
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


def on_view_mode_change(mode: str) -> tuple[dict, dict]:
    """Radio toggle between rendered markdown and raw markdown source.

    CRITICAL: ``final_summary_source`` must never be flipped to
    ``visible=False`` or the Copy button's JS loses the <textarea> it
    reads from. "Rendered" mode uses ``visible="hidden"`` (CSS-hidden but
    in DOM); "Raw" mode uses ``visible=True``.
    """
    if mode == "Raw":
        return (
            gr.update(visible=False),      # final_summary_md hidden
            gr.update(visible=True),        # final_summary_source visible
        )
    return (
        gr.update(visible=True),            # final_summary_md visible
        gr.update(visible="hidden"),        # final_summary_source hidden in DOM
    )


def on_stop(state_val: dict[str, Any]) -> tuple:
    """Stop-button click handler. 12-tuple matching the Run outputs."""
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
        gr.update(visible=True),                          # console_group
        gr.update(value=_progress_value(header, pct)),    # progress_label
        gr.update(value=log_text),                        # log_panel
        gr.update(visible=False),                         # summary_section
        gr.update(value="Rendered", visible=False),       # view_mode
        gr.update(value="", visible=False),               # final_summary_md
        gr.update(value="", visible="hidden"),            # final_summary_source (stays "hidden")
        gr.update(visible=False),                         # copy_btn
        gr.update(value=None, visible=False),             # download_btn
        gr.update(interactive=True),                      # run_btn
        state_val,                                        # session_state
        gr.update(visible=False),                         # stop_btn
    )


# 9-entry reset tuple for the nine run-output components. Used by
# on_file_upload to clear state on new upload / upload clear. Order MUST
# match the event-listener outputs list in build_demo.
#
# final_summary_source uses value + visible="hidden" explicitly — crucial
# because this reset may fire AFTER the user has toggled to Raw view
# (which sets visible=True). Without this explicit "hidden", the raw
# textbox would remain visible on the next run's initial state.
_RUN_OUTPUT_RESET = (
    gr.update(visible=False),                           # console_group
    gr.update(value={}),                                # progress_label
    gr.update(value=""),                                # log_panel
    gr.update(visible=False),                           # summary_section
    gr.update(value="Rendered", visible=False),        # view_mode
    gr.update(value="", visible=False),                 # final_summary_md
    gr.update(value="", visible="hidden"),              # final_summary_source
    gr.update(visible=False),                           # copy_btn
    gr.update(value=None, visible=False),               # download_btn
)


def on_file_upload(
    uploaded_file: str | None,
    state_val: dict[str, Any],
) -> tuple:
    """Handle file upload / clear. 16-tuple return.

    Order:
        preview_md, error_md, run_btn, all_speakers_state (was
        detected_speakers_state), speaker_map_state, session_state,
        meta_md, + 9 entries from _RUN_OUTPUT_RESET.

    Grew from 14 → 16 at M6.5 round 2 (console + summary panel).
    detected_speakers_state was renamed all_speakers_state at round 3
    (now carries [(name, is_generic), …] instead of [str, …] so the
    render function can show generic + pre-named in one pass).
    """
    if uploaded_file is None:
        state_val["canonical_md"] = None
        state_val["uploaded_stem"] = None
        state_val["final_summary_path"] = None
        state_val["log_text"] = ""
        state_val["progress_pct"] = 0
        state_val["progress_phase"] = ""
        return (
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(interactive=False),
            [],                                       # all_speakers_state
            {},                                       # speaker_map_state
            state_val,
            gr.update(value="", visible=False),       # meta_md
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
    state_val["log_text"] = ""
    state_val["progress_pct"] = 0
    state_val["progress_phase"] = ""

    all_speakers = detect_all_speakers(canonical_md)
    generic_count = sum(1 for _, is_gen in all_speakers if is_gen)

    n_turns = sum(
        1 for line in canonical_md.splitlines() if line.startswith("**")
    )
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
        all_speakers,                                 # all_speakers_state
        {},                                           # reset speaker_map
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
    """Run steps 2 → 5. 12-tuple yields matching the Run outputs list.

    Step-to-model mapping:
      Step 2 (clean_transcript)    → editor_model
      Step 3 (apply_speaker_mapping) → pure, no model
      Step 4 (extract_information) → extractor_model
      Step 5 (format_summary)      → extractor_model  (M6.5 round 2 change)

    Streaming log capture (M6.5 round 3):
      Each Ollama-touching step runs in a daemon thread via
      ``_stream_step_into_state``. The main generator polls the captured
      stdout every LOG_POLL_INTERVAL seconds and yields a fresh run-output
      tuple whenever new content is captured, so the user sees the CLI-
      equivalent prints live as they happen (rather than bursting in at
      step-end). Cancellation still has the non-streaming-Ollama delay
      (spec F3) — the main thread returns to idle immediately but the
      step thread's Ollama call runs to completion in the background
      before the model actually unloads.
    """
    # Helper builds the 12-tuple from current state_val. Closes over
    # state_val so callers don't have to restate every field.
    def _running_tuple(stop_visible: bool = True) -> tuple:
        return (
            gr.update(visible=True),                                         # console_group
            gr.update(value=_progress_value(
                state_val["progress_phase"] + "…", state_val["progress_pct"]
            )),                                                              # progress_label
            gr.update(value=state_val["log_text"]),                          # log_panel
            gr.update(visible=False),                                        # summary_section
            gr.update(value="Rendered", visible=False),                      # view_mode
            gr.update(value="", visible=False),                              # final_summary_md
            gr.update(value="", visible="hidden"),                           # final_summary_source
            gr.update(visible=False),                                        # copy_btn
            gr.update(value=None, visible=False),                            # download_btn
            gr.update(interactive=False),                                    # run_btn
            state_val,
            gr.update(visible=stop_visible),                                 # stop_btn
        )

    # ── Pre-flight: no transcript ────────────────────────────────────────
    if not state_val.get("canonical_md"):
        yield (
            gr.update(visible=True),
            gr.update(value=_progress_value("Cannot run", 0)),
            gr.update(value="❌ No transcript ingested. Upload a file first."),
            gr.update(visible=False),
            gr.update(value="Rendered", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible="hidden"),
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
            gr.update(value="Rendered", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible="hidden"),
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

    try:
        # ── Step 1/4: clean_transcript (editor_model, streamed) ──────────
        state_val["progress_pct"] = 0
        state_val["progress_phase"] = "Step 1/4 · Cleaning transcript"
        yield _running_tuple()

        for _ in _stream_step_into_state(
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

        for _ in _stream_step_into_state(
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

        for _ in _stream_step_into_state(
            state_val,
            "_final_path",
            lambda: format_summary(
                extracted_path,
                tempdir / "final",
                extractor_model,                   # step 5 uses extractor (M6.5 round 2)
                ollama_host,
            ),
        ):
            yield _running_tuple()
        final_path = state_val["_final_path"]

        # Rename the summary for download using the original upload's stem.
        stem = state_val.get("uploaded_stem") or final_path.stem
        download_dir = tempdir / "download"
        download_dir.mkdir(parents=True, exist_ok=True)
        download_path = download_dir / f"{stem}_summary.md"
        shutil.copy(final_path, download_path)
        state_val["final_summary_path"] = str(download_path)

        final_content = download_path.read_text(encoding="utf-8")

        # ── Terminal success ─────────────────────────────────────────────
        # All six summary-region components explicitly set: summary_section
        # visible, view_mode reset to "Rendered" + visible, final_summary_md
        # shown, final_summary_source filled AND visibility explicitly set
        # to "hidden". That last one matters: if the user had toggled to
        # "Raw" on a previous run, the source would still be visible=True;
        # the explicit "hidden" here resets it so only the rendered view
        # shows on the new success — no more double display (M6.5 round 3
        # bug fix).
        state_val["progress_pct"] = 100
        state_val["progress_phase"] = "Done"
        yield (
            gr.update(visible=True),                                        # console_group
            gr.update(value=_progress_value("✅ Done", 100)),               # progress_label
            gr.update(value=state_val["log_text"]),                         # log_panel
            gr.update(visible=True),                                        # summary_section
            gr.update(value="Rendered", visible=True),                      # view_mode shown + reset
            gr.update(value=final_content, visible=True),                   # final_summary_md
            gr.update(value=final_content, visible="hidden"),               # final_summary_source (back to "hidden")
            gr.update(visible=True),                                        # copy_btn
            gr.update(value=str(download_path), visible=True),              # download_btn
            gr.update(interactive=True),                                    # run_btn
            state_val,
            gr.update(visible=False),                                       # stop_btn
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
            gr.update(value="Rendered", visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible="hidden"),
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(interactive=True),
            state_val,
            gr.update(visible=False),
        )

    finally:
        state_val["run_in_progress"] = False
        for m in list(state_val.get("models_used", ())):
            unload_model(ollama_host, m)


# ─── UI construction ──────────────────────────────────────────────────────

def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks app. Layout L-C: preview + speakers
    side-by-side, everything else full-width below."""
    with gr.Blocks(title="Local Meeting Summarizer") as demo:
        session_state = gr.State(
            value=init_session_state(),
            delete_callback=cleanup_session,
        )
        # Renamed from detected_speakers_state (was list[str] of generic
        # names only). Now carries [(name, is_generic), …] so the render
        # function can show pre-named speakers disabled alongside
        # editable generic ones.
        all_speakers_state = gr.State([])
        speaker_map_state = gr.State({})

        # ── Sidebar ──────────────────────────────────────────────────────
        with gr.Sidebar():
            gr.Markdown("### Settings")

            # Host input — stacked layout for reliable alignment in a
            # narrow sidebar. Bold label, then the textbox on its own
            # full-width row (allows URL to wrap without misaligning
            # adjacent controls), then a compact row for [🔄 refresh] +
            # [colored-dot LED indicator]. Earlier attempts had all three
            # in one row, which broke visually when the URL wrapped to
            # two lines (M6.5 round 3 fix).
            gr.Markdown("**Ollama host**")
            ollama_host = gr.Textbox(
                value=DEFAULT_OLLAMA_HOST,
                placeholder="http://<host>:<port>",
                show_label=False,
                container=False,
            )
            with gr.Row(elem_id="sidebar-host-row"):
                test_btn = gr.Button(
                    "🔄",
                    scale=0,
                    min_width=40,
                    variant="secondary",
                    elem_id="test-btn",
                )
                # LED as a CSS-coloured circle inside gr.HTML. Not an
                # emoji — 🟢 rendered as a square in the user's macOS
                # testing. See _connection_indicator_html docstring.
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

            gr.Markdown("# Local Meeting Summarizer")
            gr.Markdown(
                "Upload an exported meeting transcript (`.rtf`) from "
                "[moonshine-notetaker](https://note-taker.moonshine.ai/) or "
                "from the zz's local transcriber (`.md`). Files above 10 MB "
                "are rejected; ~2.5h of speech is the practical ceiling "
                "(LLM context window)."
            )

            with gr.Accordion("Good to know", open=True):
                gr.Markdown(
                    "Each tab runs independently — multiple tabs = multiple "
                    "queue slots. When you close the tab, the models your "
                    "session loaded and its temp files are cleaned up "
                    "automatically (within about an hour)."
                )

            # ── Upload (full width, above the split row) ─────────────────
            upload = gr.File(
                label="Transcript",
                file_types=[".rtf", ".md"],
                file_count="single",
                elem_id="transcript-upload",
            )

            error_md = gr.Markdown("", visible=False, elem_id="ingest-error")
            meta_md = gr.Markdown("", visible=False, elem_id="ingest-meta")

            # ── L-C split: Preview (left) + Speakers (right) ─────────────
            # Two equal-width columns. Works because the page is wider
            # than a laptop sidebar and these two components benefit from
            # being cross-referenced (read the transcript → type the
            # name). Console and Summary stay full-width below because
            # they're heavier content that needs the whole page.
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
                    # @gr.render rebuilds its children each time its
                    # inputs change. ``all_speakers`` is now a list of
                    # (name, is_generic) tuples — we render ALL speakers,
                    # not just the generic ones. Pre-named speakers get
                    # interactive=False with value=name so the form shows
                    # a consistent view of every speaker in the
                    # transcript. User can't accidentally re-type a
                    # name that was already set upstream (M6.5 round 3).
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
                                )
                            else:
                                # Pre-named: disabled textbox showing the
                                # existing name. Not connected to any
                                # change handler — the pipeline already
                                # has the right name in the transcript.
                                gr.Textbox(
                                    label=f"{name} (already named)",
                                    value=name,
                                    interactive=False,
                                )

            # ── Run + Stop row (full width below the split) ──────────────
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

            # ── Console panel: progress (gr.Label) + log ─────────────────
            with gr.Group(visible=False) as console_group:
                # gr.Label replaces the hand-rolled gr.HTML progress bar.
                # Value is dict[str, float]; gr.Label renders as a labelled
                # bar with percentage, using the theme's accent colour
                # automatically. show_heading=False hides the big
                # duplicated argmax label — we only have one entry anyway,
                # and the bar's inline label is enough.
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

            # ── Results section (separate panel, full width) ─────────────
            with gr.Column(variant="panel", visible=False) as summary_section:
                # Radio toggle between rendered markdown and raw source.
                # Visibility driven explicitly from run_pipeline_generator
                # yields — we do NOT rely on on_view_mode_change firing
                # when the value is set programmatically (Gradio .change()
                # only fires on user interaction). See the success yield
                # for the explicit final_summary_source visibility reset.
                view_mode = gr.Radio(
                    choices=["Rendered", "Raw"],
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
                    visible=False,
                    elem_id="final-summary",
                )

                # See COPY_SUMMARY_JS + on_view_mode_change for why
                # visibility toggles between "hidden" (default) and True
                # (Raw mode) — NEVER False, which would break Copy.
                final_summary_source = gr.Textbox(
                    value="",
                    visible="hidden",
                    elem_id="final-summary-source",
                    interactive=True,
                    show_label=False,
                    lines=SUMMARY_RAW_LINES,
                    max_lines=SUMMARY_RAW_LINES,
                )

                with gr.Row():
                    # Labels compacted to "Copy" / "Download" per user
                    # request (the component context makes the verb
                    # sufficient; no need for "summary" / ".md").
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

        # ── Event wiring ──────────────────────────────────────────────────

        demo.load(
            on_startup,
            inputs=[session_state],
            outputs=[banner, editor_status, extractor_status, connection_indicator],
        )
        demo.load(fn=None, inputs=None, outputs=None, js=FORCE_DARK_MODE_JS)

        ollama_host.change(
            on_host_change,
            inputs=[ollama_host, editor_model, extractor_model, session_state],
            outputs=[session_state, banner, editor_status, extractor_status, connection_indicator],
        )

        editor_model.change(
            _model_indicator,
            inputs=[ollama_host, editor_model],
            outputs=[editor_status],
        )
        extractor_model.change(
            _model_indicator,
            inputs=[ollama_host, extractor_model],
            outputs=[extractor_status],
        )

        test_btn.click(
            on_test_connection,
            inputs=[ollama_host],
            outputs=[connection_indicator, banner],
        )

        view_mode.change(
            on_view_mode_change,
            inputs=[view_mode],
            outputs=[final_summary_md, final_summary_source],
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
                console_group,
                progress_label,
                log_panel,
                summary_section,
                view_mode,
                final_summary_md,
                final_summary_source,
                copy_btn,
                download_btn,
            ],
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
        )

        stop_btn.click(
            on_stop,
            inputs=[session_state],
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
            cancels=[run_event],
        )

        copy_btn.click(
            None,
            inputs=None,
            outputs=None,
            js=COPY_SUMMARY_JS,
        )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────

def main() -> None:
    if not DEFAULT_OLLAMA_HOST:
        print(
            "Error: OLLAMA_HOST is missing. Please define it in a .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    _install_process_hooks()

    demo = build_demo()
    demo.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        inbrowser=True,
        theme=gr.Theme.from_hub("Nymbo/Nymbo_Theme"),
        css=CUSTOM_CSS,
        max_file_size=UPLOAD_MAX_SIZE,
    )


if __name__ == "__main__":
    main()
