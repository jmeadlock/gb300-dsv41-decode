#!/usr/bin/env python3
"""sidecar_watchdog_p3.py — fail-closed guard for the Garner-tree lane (peer tier v3 for T <= 64, v2/b12x above).

Clone of sidecar_watchdog.py v2 (production h2-REF), extended for peer tier v3. Production's watchdog is not touched.

Why: on a peer tier v3 wait timeout (PEER3_SPINS polls, his default 20,000,000) all three v3 wait sites -- peer3
finish_kernel, the boundary fold, and the MegaMoE fork's W6 fold -- SKIP the call's cold-expert adds and keep going:
tokens stream, silently wrong, and the timeout is counted only in GB300 device memory. (v2 is different: his
peer_tier2._wait bumps words[2] and then TRAPS, so v2 already fails loud.) The lane must therefore run with
PEER3_SPINS effectively unbounded (>= 1e12, plain digits -- fastpath_peer3 parses it with int()), so a dead or late
sidecar HANGS the GB300 instead of corrupting tokens, and this watchdog turns the hang into a fail-stop in seconds.
It REFUSES TO ARM (exit 3) unless `docker inspect <server>` shows that. Set it with DS41_PEER3_SPINS in the pol tree.

Reads every --interval s:
  v2 /dev/shm/vllm_peer_tier2  int64 words [0] published seq, [1] completed seq, [2] timeouts
  v3 /dev/shm/vllm_peer_tier3  u64 @128 done tag (RTX), word0 of rec[0] @4096 (>>48 = tag the GB300 last published),
                               u64 @512 RTX calls, u64 @64 protocol version
Trips when (server running):
  v2 words[2] increases | v2 published != completed with completed frozen for > --stall s |
  v3 published tag != done tag with both frozen for > --stall s |
  sidecar container not running | either segment unreadable.
On trip (--mode enforce): docker kill <server>, docker stop -t 5 <sidecar> (kill if that fails). Fail-stop: no rollback
unless --rollback is given (same code path as v2). Exits 2 after a trip; 3 if the spin preconditions do not hold.
stdlib only.
"""
import argparse, json, os, struct, subprocess, sys, time, urllib.request

SHM2 = "/dev/shm/vllm_peer_tier2"
SHM3 = "/dev/shm/vllm_peer_tier3"
OFF_VERSION, OFF_DONE, OFF_RTXSTATS, OFF_REC = 64, 128, 512, 4096
GB300 = "GPU-1c1855c8-dd10-0247-583e-3c3880ddbf55"
MIN_V3_SPINS = 10 ** 12


def words2(path):
    with open(path, "rb") as f:
        return struct.unpack("<3q", f.read(24))


def seg3(path):
    """(version, done_tag, published_tag, rtx_calls) from the v3 segment."""
    with open(path, "rb") as f:
        b = f.read(OFF_REC + 8)
    u64 = lambda o: struct.unpack_from("<Q", b, o)[0]
    return u64(OFF_VERSION), u64(OFF_DONE) & 0xFFFF, (u64(OFF_REC) >> 48) & 0xFFFF, u64(OFF_RTXSTATS)


def sh(args, timeout=10):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return None, ""


def running(name):
    rc, out = sh(["docker", "inspect", "-f", "{{.State.Running}}", name])
    if rc is None:
        return None
    return rc == 0 and out == "true"


def container_env(name):
    rc, out = sh(["docker", "inspect", "-f", "{{json .Config.Env}}", name])
    if rc != 0:
        return None
    try:
        return dict(e.split("=", 1) for e in json.loads(out) if "=" in e)
    except Exception:
        return None


def spin_preconditions(env):
    """[] when the v3 wait is effectively unbounded, else the list of problems."""
    if env is None:
        return ["cannot read server env"]
    raw = env.get("PEER3_SPINS")
    if raw is None:
        return ["PEER3_SPINS unset (his default 20000000: a timeout SKIPS the cold add => wrong tokens)"]
    if not raw.isdigit():
        return [f"PEER3_SPINS={raw!r} is not plain digits (fastpath_peer3 int() would fail at boot)"]
    if int(raw) < MIN_V3_SPINS:
        return [f"PEER3_SPINS={raw} < {MIN_V3_SPINS} (a timeout SKIPS the cold add => wrong tokens)"]
    return []


def log(msg, fh):
    line = f"{time.strftime('%F %T %Z')} {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def fail_closed(a, side, fh):
    rc, _ = sh(["docker", "kill", a.server], timeout=30)
    log(f"fail-closed: docker kill {a.server} rc={rc}", fh)  # the server kill is what stops wrong/stalled output
    sh(["docker", "unpause", side], timeout=10)              # a frozen (paused) sidecar blocks docker stop
    rc2, _ = sh(["docker", "stop", "-t", "5", side], timeout=30)
    if rc2 != 0:
        rc2, _ = sh(["docker", "kill", side], timeout=30)
    log(f"fail-closed: sidecar {side} stop/kill rc={rc2}", fh)
    if not a.rollback:
        log("no --rollback: lane left DOWN (fail-stop)", fh)
        return
    rc, _ = sh(["docker", "start", a.rollback], timeout=60)
    log(f"rollback docker start {a.rollback} rc={rc}", fh)


class State:
    """Pure stall/timeout logic, separated from I/O so it can be unit-tested."""

    def __init__(self, stall):
        self.stall = stall
        self.base2 = None
        self.last2 = None
        self.lag2 = None
        self.last3 = None
        self.lag3 = None

    def step(self, now, w2, s3):
        reason = None
        if w2 is not None:
            if self.base2 is None or w2[2] < self.base2:
                self.base2 = w2[2]
            if w2[2] > self.base2:
                reason = f"v2 sidecar TIMEOUTS {self.base2}->{w2[2]} (cold contribution dropped => wrong tokens)"
            if w2[0] != w2[1] and self.last2 is not None and w2[1] == self.last2:
                self.lag2 = self.lag2 or now
                if now - self.lag2 > self.stall:
                    reason = reason or f"v2 stalled: published {w2[0]} completed {w2[1]} for >{self.stall}s"
            else:
                self.lag2 = None
            self.last2 = w2[1]
        if s3 is not None:
            _, done, pub, calls = s3
            key = (done, pub)
            if pub != 0 and pub != done and self.last3 == key:
                self.lag3 = self.lag3 or now
                if now - self.lag3 > self.stall:
                    reason = reason or (f"v3 stalled: GB300 published tag {pub:#06x}, RTX done {done:#06x}, "
                                        f"rtx calls {calls}, frozen for >{self.stall}s")
            else:
                self.lag3 = None
            self.last3 = key
        return reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--sidecar", default=None, help="default: <server>-sidecar")
    ap.add_argument("--rollback", default=None)
    ap.add_argument("--mode", choices=["enforce", "report"], default="enforce")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--stall", type=float, default=2.0)
    ap.add_argument("--shm2", default=SHM2)
    ap.add_argument("--shm3", default=SHM3)
    ap.add_argument("--status", default="/home/milo/dsv41/m3/logs/sidecar_watchdog_p3.status.json")
    ap.add_argument("--log", default="/home/milo/dsv41/m3/logs/sidecar_watchdog_p3.log")
    ap.add_argument("--skip-env-check", action="store_true", help="tests only")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    side = a.sidecar or a.server + "-sidecar"
    fh = open(a.log, "a")
    if not a.skip_env_check:
        bad = spin_preconditions(container_env(a.server))
        if bad:
            for b in bad:
                log(f"REFUSE TO ARM: {b}", fh)
            return 3
    st = State(a.stall)
    log(f"start p3 mode={a.mode} server={a.server} sidecar={side} stall={a.stall}s interval={a.interval}s", fh)
    while True:
        reason = None
        srv_up = running(a.server)
        w2 = s3 = None
        try:
            w2 = words2(a.shm2)
        except Exception as e:
            if srv_up:
                reason = f"v2 shm unreadable while server runs: {e}"
        try:
            s3 = seg3(a.shm3)
        except Exception as e:
            if srv_up:
                reason = reason or f"v3 shm unreadable while server runs: {e}"
        reason = st.step(time.time(), w2, s3) or reason
        if srv_up and running(side) is False:
            reason = reason or f"sidecar container {side} not running while {a.server} serves"
        if srv_up is None:
            log("WARN docker inspect timed out", fh)
        status = {"t": time.time(), "server_up": bool(srv_up), "v2": w2, "v3": s3, "timeouts_base": st.base2,
                  "mode": a.mode, "trip": reason, "stall_s": a.stall, "v": "p3"}
        try:
            with open(a.status + ".tmp", "w") as f:
                json.dump(status, f)
            os.replace(a.status + ".tmp", a.status)
        except Exception:
            pass
        if a.once:
            print(json.dumps(status)); return 0
        if reason and srv_up:
            log(f"TRIP {reason}", fh)
            if a.mode == "enforce":
                fail_closed(a, side, fh)
                return 2
            st = State(a.stall)
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
