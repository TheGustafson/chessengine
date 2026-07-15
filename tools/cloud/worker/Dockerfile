FROM python:3.12-slim-trixie@sha256:c3d81d25b3154142b0b42eb1e61300024426268edeb5b5a26dd7ddf64d9daf28

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/archon/bin:${PATH}

WORKDIR /work

COPY requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock

COPY bin/ /opt/archon/bin/
COPY nets/ /opt/archon/nets/
COPY tools/ /opt/archon/tools/
COPY provenance.json /opt/archon/provenance.json

RUN chmod 0555 /opt/archon/bin/* /opt/archon/tools/*.py \
    && chmod 0444 /opt/archon/nets/* /opt/archon/provenance.json

LABEL org.opencontainers.image.title="Archon research worker" \
      org.opencontainers.image.description="Pinned ARCHON, Reckless, Stockfish 18, and oracle-labeling tools"

ENTRYPOINT ["archon-worker"]
CMD ["verify"]
