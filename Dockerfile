FROM python:3.11-slim

# Fixed non-root user; for bind mounts run with --user "$(id -u):$(id -g)"
RUN useradd -m -u 1000 app && install -d -o app -g app /app

WORKDIR /app

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY --chown=app:app visualmic.py VERSION ./
COPY --chown=app:app tests/ tests/
COPY --chown=app:app scripts/ scripts/

ARG VERSION
LABEL version=${VERSION}

USER app

ENTRYPOINT ["python", "-u", "visualmic.py"]
