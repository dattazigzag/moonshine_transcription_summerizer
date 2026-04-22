FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Install deps + build project.
# README.md is required because pyproject.toml sets `readme = "README.md"`
# and hatchling reads it at sync time. Same gotcha as our other two
# projects — don't strip it from .dockerignore.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

COPY app.py main.py ./
COPY pipeline/ pipeline/
COPY assets/ assets/

# --------------------------------------------------------------------
# Runtime environment
# --------------------------------------------------------------------

# Unbuffered stdout so `docker logs -f` shows pipeline progress in real
# time instead of after each step flushes. Required for the Gradio
# streaming log panel to also feel live when watched from the host.
ENV PYTHONUNBUFFERED=1

# Gradio behaviour: don't try to send anonymous usage pings, and don't
# auto-open a browser (app.py already passes inbrowser=False, but this
# is belt-and-braces for any future launch path or downstream image).
ENV GRADIO_ANALYTICS_ENABLED=False
ENV GRADIO_DO_NOT_TRACK=True

EXPOSE 2070

CMD ["uv", "run", "python", "app.py", "--port", "2070"]
