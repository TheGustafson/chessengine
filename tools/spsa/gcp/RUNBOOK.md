# Distributed SPSA on GCP spot — runbook (no OpenBench)

Status: STAGED, not runnable — gcloud is GET-only for the agent (user
mandate). Everything below is copy-paste for when the user lifts it
(scoped: one bucket + N spot VMs).

## Shape

Coordinator (spsa_dist.py) runs ON THE LOCAL BOX — the brain never
leaves home. Workers are stateless spot VMs running worker.py against a
gs:// exchange. Fixed-node games ⇒ heterogeneous/preempted workers add
zero measurement noise; preempted units are auto-reissued after
--unit-timeout. Local + cloud workers mix freely on the same exchange.

Cost math (stage-B scale): 40k games ≈ 800 CPU-h ≈ $10-20 spot
(e2-highcpu-16 spot ≈ $0.13/h ≈ 16 workers/VM). 6 such VMs ≈ 96
workers ≈ done in ~2-3h wall.

## One-time setup (user or unlocked agent)

    gcloud storage buckets create gs://archon-spsa --location=us-central1

## Per-campaign

    # 1. Stage the spec (binary+fastchess+book+net) from the local box:
    python3 tools/spsa/spsa_dist.py --config tools/spsa/campaigns/s2-stageA.json \
      --engine engines/bin/archon-d47v-tune --eval-file /tmp/d47v-a4b2.cenn \
      --exchange gs://archon-spsa/stageA --out /tmp/ignore --stage-only

    # 2. Spot workers (16 workers per e2-highcpu-16):
    gcloud compute instance-templates create spsa-worker \
      --machine-type=e2-highcpu-16 --provisioning-model=SPOT \
      --image-family=debian-12 --image-project=debian-cloud \
      --instance-termination-action=DELETE \
      --metadata=exchange=gs://archon-spsa/stageA,nworkers=16 \
      --metadata-from-file=startup-script=tools/spsa/gcp/worker_startup.sh
    gcloud compute instance-groups managed create spsa-mig \
      --template=spsa-worker --size=6 --zone=us-central1-a

    # 3. Coordinator, local box (same command as a local run, workers=0):
    bash tools/spsa/run_spsa_dist.sh tools/spsa/campaigns/s2-stageA.json \
      gs://archon-spsa/stageA engines/bin/archon-d47v-tune /tmp/d47v-a4b2.cenn 0

    # 4. Teardown (coordinator writes control/STOP at DONE; workers
    #    self-exit after 30 min idle; then):
    gcloud compute instance-groups managed delete spsa-mig --zone=us-central1-a -q

## Notes

- Workers need only python3 + gcloud (both in the debian-12 image).
- The engine binary must target the worker CPU: build the STAGED binary
  with x86-64-v3 (NOT native) — spot fleets are heterogeneous.
  PGO+v3: `PGO_CPU=x86-64-v3 bash tools/build_pgo.sh sota,arch-d47v,tune ...`
- Service account needs objectAdmin on the bucket only.
- Verify gate after any tune, per protocol: 10+0.1 SPRT vs untuned on
  the local box; SPSA output never lands in params.rs without it.
