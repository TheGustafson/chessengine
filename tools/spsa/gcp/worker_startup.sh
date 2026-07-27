#!/usr/bin/env bash
# Spot-VM startup: fetch worker.py + exchange.py from the exchange's spec/
# and run N workers (one per vCPU). Exchange URI + worker count come from
# instance metadata. Stateless; preemption-safe by design.
set -eu
EX=$(curl -s -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/attributes/exchange')
N=$(curl -s -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/attributes/nworkers')
HOST=$(hostname)
mkdir -p /opt/spsa && cd /opt/spsa
gcloud storage cp -q "$EX/spec/worker.py" "$EX/spec/exchange.py" .
for i in $(seq 1 "$N"); do
  nohup python3 worker.py --exchange "$EX" --id "$HOST-$i" \
    --cache "/tmp/spsa-cache-$i" > "/var/log/spsa-worker-$i.log" 2>&1 &
done
