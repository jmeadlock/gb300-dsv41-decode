#!/bin/bash
# lease.sh -- LOOSE-ENDS-2 (Astra review, accepted): ONE box lease. Every box script SOURCES this first:
#
#     source "$(dirname "$0")/lease.sh"        # blocks until it holds the lease (flock -w 36000 = up to 10 h)
#     ... docker run ... bench/headline.py ...  # only one lease holder runs at a time
#
# Why: the 2026-11-07 box rounds raced on one checkout and one GPU (parallel waiters each `git checkout`ed their
# branch; a PDL probe ran concurrently with a headline). A docker-ps guard missed unnamed containers. A kernel
# advisory lock cannot be fooled by naming.
#
# What it does
#   - lock file  ${DSV41_BOX_LOCK:-$HOME/dsv41/box.lock}   (the handoff's ~/dsv41/box.lock)
#   - fd 9 is opened on it and `flock -w 36000 9` taken. The lock lives as long as that open file description:
#     it is released when the script exits (normally, by `exit`, by a signal, or by `kill -9`) -- the kernel closes
#     fd 9. There is NO stale-lock state to clean up and nothing to "fail closed" about a dead holder: a dead holder
#     has already released it. (Do not rm the lock file while anyone holds it: the next opener would lock a new inode.)
#   - One caveat, by design: a child that inherits fd 9 and outlives the script (nohup'ed, setsid, a daemon) keeps
#     the lease. A `docker run` container does not inherit it. Launch detached work with `9>&-` to give the lease
#     back to the script's life (`nohup cmd 9>&- &`), or leave it inherited if the detached job IS the lease's work.
#   - re-entrant: a script that sources lease.sh while an ancestor already holds the lease (BOX_LEASE_HELD=1 in the
#     environment, set below and inherited) does nothing.
#   - timeout (36000 s) or no flock binary: prints why and exits 75 (EX_TEMPFAIL) -- fail closed, never run unleased.
#     DSV41_BOX_LEASE_WAIT overrides the wait (seconds; the handoff default is 36000).
#   - ${LOCK}.holder (informational only, never trusted): who took it, when, what. Waiters print it while they wait.
#   - exports for the receipt (bench/receipt.py): RECEIPT_LEASE, RECEIPT_GIT_COMMIT/BRANCH/DIRTY_SHA256/DIRTY_FILES/
#     UNTRACKED -- computed on the HOST from $BOX_REPO (default: $HOME/dsv41/decode), because the vllm container has
#     no git / a 'dubious ownership' checkout. Pass them into docker with `$(box_lease_docker_env)`:
#         D="docker run --rm ... $(box_lease_docker_env) ..."
#   - works under `set -u` and `set -e` (the existing scripts use `set -u`).
#
# Adoption (not done here: the existing scripts are not this worker's files): add the `source` line above as the first
# line after `set -u`, and drop the `while docker ps ...; do sleep 20; done` guard (keep it only as belt-and-braces
# for manual runs that do not source this file).

BOX_LOCK="${DSV41_BOX_LOCK:-$HOME/dsv41/box.lock}"
BOX_LEASE_WAIT="${DSV41_BOX_LEASE_WAIT:-36000}"
BOX_REPO="${BOX_REPO:-$HOME/dsv41/decode}"

_box_lease_who() {   # one line: "<user>@<host> pid <pid> since <iso> <script>"
    printf '%s@%s pid %s since %s %s' "${USER:-?}" "$(hostname 2>/dev/null || echo ?)" "$$" \
        "$(date '+%Y-%m-%d %H:%M:%S %Z')" "${0:-?}"
}

_box_lease_token() {  # the same, space-free (safe inside an unquoted `docker run -e K=V` string)
    printf '%s@%s:pid=%s:since=%s:%s' "${USER:-?}" "$(hostname 2>/dev/null || echo ?)" "$$" \
        "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$(basename "${0:-?}" | tr ' ' '_')"
}

_box_git() { git -c 'safe.directory=*' -C "${BOX_REPO}" "$@"; }

box_lease_receipt_env() {   # exports RECEIPT_* (host-side git state for the container's receipt); never fails
    export RECEIPT_LEASE="${BOX_LOCK}|$(_box_lease_token)"
    if command -v git >/dev/null 2>&1 && [ -d "${BOX_REPO}" ]; then
        export RECEIPT_GIT_COMMIT="$(_box_git rev-parse HEAD 2>/dev/null || true)"
        export RECEIPT_GIT_BRANCH="$(_box_git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
        local d; d="$(_box_git diff HEAD --binary 2>/dev/null | { sha256sum 2>/dev/null || shasum -a 256 2>/dev/null; } | cut -d' ' -f1)"
        # sha256 of empty input = e3b0c442...: report "" (clean) in that case, to match receipt.py (None when no diff)
        if [ "${d:-}" = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ]; then d=""; fi
        export RECEIPT_GIT_DIRTY_SHA256="${d:-}"
        export RECEIPT_GIT_DIRTY_FILES="$(_box_git diff HEAD --name-only 2>/dev/null | tr ' ' '_' | paste -sd, - || true)"
        export RECEIPT_GIT_UNTRACKED="$(_box_git ls-files --others --exclude-standard 2>/dev/null | wc -l | tr -d ' ' || true)"
    fi
    return 0
}

box_lease_docker_env() {    # prints "-e RECEIPT_X=..." for docker run (values are space-free by construction: commit, sha, csv names with spaces -> _)
    local v out=""
    for v in RECEIPT_LEASE RECEIPT_GIT_COMMIT RECEIPT_GIT_BRANCH RECEIPT_GIT_DIRTY_SHA256 RECEIPT_GIT_DIRTY_FILES RECEIPT_GIT_UNTRACKED; do
        if [ -n "${!v:-}" ]; then out="$out -e $v=${!v}"; fi
    done
    printf '%s' "$out"
}

box_lease_acquire() {
    if [ "${BOX_LEASE_HELD:-}" = "1" ]; then return 0; fi          # an ancestor holds it (fd 9 is inherited)
    if ! command -v flock >/dev/null 2>&1; then
        echo "lease.sh: no flock binary on this host -- refusing to run unleased (exit 75)" >&2
        return 75
    fi
    mkdir -p "$(dirname "${BOX_LOCK}")" || return 75
    exec 9>>"${BOX_LOCK}" || { echo "lease.sh: cannot open ${BOX_LOCK}" >&2; return 75; }
    if ! flock -n 9; then
        echo "lease.sh: box lease held -- $(cat "${BOX_LOCK}.holder" 2>/dev/null || echo 'holder unknown'); waiting up to ${BOX_LEASE_WAIT}s ($(date +%H:%M))" >&2
        if ! flock -w "${BOX_LEASE_WAIT}" 9; then
            echo "lease.sh: gave up after ${BOX_LEASE_WAIT}s waiting for ${BOX_LOCK} (holder: $(cat "${BOX_LOCK}.holder" 2>/dev/null || echo unknown))" >&2
            return 75
        fi
    fi
    export BOX_LEASE_HELD=1
    _box_lease_who > "${BOX_LOCK}.holder" 2>/dev/null || true
    box_lease_receipt_env
    echo "lease.sh: box lease taken ($(_box_lease_who))" >&2
    return 0
}

box_lease_acquire || exit $?
