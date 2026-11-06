# Cutover: Station becomes the dev box — checklist (reversible)

Status: **EXECUTED 2026-11-06 10:45–10:50 CST** (James "go"). Done: consumers repointed (Milo delegation deep/light → `custom[dsv41-spark]`; Alyx `model.provider` → `custom[dsv41-spark]`; Echo `auxiliary.vision` + `vision` preset → Spark, config only — **gateway restart on Forge pending James** (`systemctl --user restart hermes-gateway` on oc-milo-ms01; Milo's tool refuses gateway restarts); gateway alias `deepseek-v4.1-flash` shelved + NOTICE + 4 guest keys trimmed (commit 22f123c); Pg61 lane-check cron paused; LANE-OWNER line + MAINTENANCE written. Smokes: Hermes one-shot tool call on Spark 7 s; gateway spark-native PONG. **Pg61 deliberately still UP for W0** (measurement needs it); step 3 runs at W0 end.

Live state checked 2026-11-06 10:37 CST: Pg61 up on `:30006` (`ds41` + `ds41-sidecar`, 3 h); Spark 1–4 SGLang up on `192.168.1.51:8888` serving `deepseek-v4.1-flash`; `custom[dsv41-spark]` already defined in milo `config.yaml`; no `MAINTENANCE` flag; root 368 G free, `/models` 1.5 T free.

## Consumers that move to Spark `:8888`

| Consumer | Today | After |
|---|---|---|
| Milo (`custom[dsv41]`, model `dsv41-flash-uva`) | Station `:30006` | `custom[dsv41-spark]` / `deepseek-v4.1-flash` |
| Milo subagents (`custom[dsv41]` per 2026-10-30 rule) | Station | kernel-design → Anthropic; desk/grunt → `custom[dsv41-spark]` |
| Echo on Forge (default since 2026-10-17) | Station | Spark `:8888` (per `lab-inference-gateway-ops`) |
| Alyx (`custom[alyx-local]`) | Station | Spark `:8888` |
| Forge guest gateway `deepseek-v4.1-flash` | Station | registry row → `deepseek-v4.1-flash-spark-native` (already serving) |
| Vision tier (milo-wow lane-a copilot) | Station ~1 s | Spark 2–4 s — **degraded**, say so |

## Steps (in order)

1. Repoint consumers above; smoke each: one Hermes one-shot with a tool call on `custom[dsv41-spark]`; Echo one chat; Forge gateway in-container curl; Alyx one chat.
2. Stop the 10-min Pg61 lane-check cron on the M4 (`dsv41_pg61_lane_check.sh`) — disable, don't delete.
3. On the box: `cd ~/dsv41/garner && WORK=~/dsv41/garner/pg61-work DS41_HOME=~/dsv41/garner/home launcher/launch-pg61.sh down` (stops+removes `ds41`/`ds41-sidecar`, stops p3 watchdog, clears peer shm).
4. Write `~/dsv41/m3/logs/MAINTENANCE` ("decode-runtime project; Station is dev box; no restore until James says") and a dated `LANE-OWNER.txt` line.
5. Verify: `nvidia-smi` both GPUs near 0 MiB; `ss -ltn | grep 30006` closed; `pgrep -af "[s]idecar_watchdog"` empty; Spark `/v1/models` 200.
6. Tell James which consumers went dark/degraded, with the 12 h ETA convention for W0.

## Rollback (one command + repoint)

`launch-pg61.sh up` with the same env plus `MODEL_DIR` (~6 min to bind), re-arm p3 watchdog ENFORCE (`PEER3_SPINS=1000000000000`), repoint consumers back, re-enable the cron, archive `MAINTENANCE`. Restore bundles from the K3 and Ash windows are on the box as templates.

## Not part of cutover

- No image pulls, no `/models` changes, no deletion of Pg61 containers' images or the Garner tree.
- RTX PRO 6000 stays idle until M3.
