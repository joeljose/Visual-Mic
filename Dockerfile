FROM python:3.11.17-slim@sha256:45037981b62b34b44602584fccbc4d884d5f7dc92c7ee86bb38a698a79fe1e51

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
