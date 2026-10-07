FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --system --uid 1000 doorstep && mkdir -p /data /config && chown doorstep /data
USER doorstep

ENV DOORSTEP_CONFIG=/config/config.yaml
VOLUME ["/data"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=4)" || exit 1
ENTRYPOINT ["doorstep"]
CMD ["serve"]
