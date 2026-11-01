FROM python:3.11.18-slim@sha256:ffe4b0c76963e64f05484d74dbb3f67fad1effcf3c5c22c2235f0fafc8a9c98f

# Fixed non-root user; for bind mounts run with --user "$(id -u):$(id -g)"
RUN useradd -m -u 1000 app && install -d -o app -g app /app

WORKDIR /app

# Hash-locked dependencies; regenerate requirements.lock as described in
# CONTRIBUTING.md after changing requirements.txt or requirements-dev.txt
COPY requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY --chown=app:app visualmic.py VERSION ./
COPY --chown=app:app tests/ tests/
COPY --chown=app:app scripts/ scripts/

ARG VERSION
LABEL version=${VERSION}

USER app

ENTRYPOINT ["python", "-u", "visualmic.py"]
