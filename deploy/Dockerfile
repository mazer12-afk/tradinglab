# TradingLab – autonomous paper-trading daemon.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY config ./config
# CPU-only torch keeps the image small.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
 && pip install -e .
COPY models ./models
VOLUME ["/app/state", "/app/data", "/app/reports"]
HEALTHCHECK --interval=10m --timeout=10s --start-period=30m \
  CMD python -c "import json,sys,time,datetime as d; h=json.load(open('/app/state/heartbeat.json')); \
age=(d.datetime.now(d.timezone.utc)-d.datetime.fromisoformat(h['ts'])).total_seconds(); sys.exit(0 if age<3600 else 1)"
ENTRYPOINT ["tradinglab"]
CMD ["run"]
