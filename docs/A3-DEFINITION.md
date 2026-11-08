# A3 — the M0 parity metric, exactly as the code computes it

LOOSE-ENDS-2 (Astra review, accepted). Every number below was recomputed from the files named, with the repo's own
`bench/parity.py` (the code that `bench/first_light.py` runs for its parity table — `run_parity()` shells out to it with
`--ref w0/results/phase1/parity-A.jsonl --ref2 .../parity-A2.jsonl --cand <rep>.jsonl --json …`). Nothing here is a
new measurement; `tests/test_receipt_cpu.py::test_a3_doc_numbers_recompute` re-derives the worked example from the
same files and fails if this page drifts. **Measured** = read from a receipt/JSONL; **derived** = arithmetic on those;
**modeled** = none.

## 1. What was approved, and what the code computes (they differ)

`PROJECT.md` A3 (approved 2026-11-07): *self-replay bit-identical; 0 margin-weighted top-1 flips and mean |Δlogprob| ≤
0.05 vs Pg61 over the shared prefix; tiered == all-HBM bit-exact; long-horizon quality deferred to M1.*

`bench/parity.py` computes, per fixture prompt, for the candidate (`cand`) against Pg61 run A (`ref`), and for Pg61 run A2
against A as the noise floor:

| field | definition in code (`compare`) |
|---|---|
| `len_ref`, `len_cand` | number of generated positions in each JSONL (`logprobs` list length) |
| `first_divergence` | smallest `i < min(len_ref, len_cand)` with `ref.token[i] != cand.token[i]`; `None` if none. **A length mismatch alone is NOT a divergence** (see §4) |
| `shared_prefix` | `first_divergence` if there is one, else `min(len_ref, len_cand)`. The prefix lengths are per prompt: 256/256 for prose, code, math, shell, long; 18 json; 21 tool; 157 story (A) |
| `top1_flips` | over `i in [0, shared_prefix)`: count of positions where `argmax(cand.top_logprobs)` ≠ `ref`'s top-1 token |
| `top1_flips_weighted` | the sum of those positions' `|margin|`, `margin = ref_top1.logprob − ref_top2.logprob` (ref's `top_logprobs` sorted descending; `0.0` if ref lists < 2). Rounded to **4** decimals |
| `max_abs_dlogprob` | `max_i |ref.logprob[i] − cand.logprob[i]|` over the same prefix — the logprob of the *sampled token* under each run. Rounded to **5** decimals |

Per-prompt pass rule (`grade`), the *pre-A3* bar, still what the printed `PARITY: PASS/FAIL` line means:
- `A2` had no divergence ⇒ `cand` must have none; if `A2` diverged at p, `cand` must not diverge before p;
- `cand.top1_flips_weighted ≤ A2.top1_flips_weighted + 1e-9`;
- `cand.max_abs_dlogprob ≤ max(A2.max_abs_dlogprob, 0.05)`  ← **max**, not mean;
- overall: every prompt ok **and** `identical ≥ 7/8` (identical = `first_divergence is None`).

So the literal table (`identical 2/8; PARITY: FAIL`) is the old bar; `LEDGER.md` 15:54 "A3 check" reads only the
`flips_w` column. Two facts about A3-as-implemented follow, both checked on the files in the tree:

1. **`top1_flips` is structurally 0 in this code.** The loop stops at the first token divergence, and before it the
   candidate's sampled token equals the reference's, which is the argmax of its own `top_logprobs` (verified:
   0 of 1,476 positions in A, 0 of 1,444 in A2, 0 of 1,501 in the cand, have token ≠ argmax(top_logprobs); k = 20 everywhere).
   So "0 margin-weighted flips" **cannot fail** as computed — the 0.000 on 8/8 prompts is not evidence of parity. A flip
   can only happen *at or after* a divergence, which the loop never looks at.
2. **The approved text says mean |Δlogprob| ≤ 0.05; the code gates the max.** They are different statistics (§3 shows both).

Neither is a bug in anyone's measurement of anything else; it means A3's top-1 clause needs the arm in §5 to carry
information. The divergence *position* and the margin there (§2) are the informative quantities the table already has.

## 2. Worked example (`w0/results/first-light-cand-night-20261107.jsonl` vs `parity-A.jsonl`)

This JSONL reproduces, digit for digit, the parity table stored in `docs/first-light-t9b4-20261107.json` and
`docs/first-light-r2-20261107.json` (`parity.1.stdout`). The T9b-8 table (`fl-t9b8`) named in `LEDGER.md` lives in the
box scratch tree and is not in this checkout, so it is not used here.

Row `code` (first divergence at position **114**):
- `len_ref = 256`, `len_cand = 256`, `shared_prefix = 114` (positions 0…113 compared);
- `top1_flips = 0`, `top1_flips_weighted = 0.0` — by §1 fact 1;
- `max_abs_dlogprob = 0.31493`, at position 15 (derived: argmax over the prefix of |Δ|); mean |Δ| over the 114 positions = **0.01910** (derived — `parity.py` does not print it);
- divergence token: ref `' ties'` (logprob −0.9007) vs cand `' stable'` (−0.9329). Ref top1−top2 margin there = **0.1250**
  (Pg61 logprobs are bf16-coarse: margins come in steps of 0.125, 0.25, 0.5, 1.0); cand's own gap between its top-1
  and ref's token = **0.0120**. A near-tie by either engine's distribution.
- verdict: A2 on `code` diverges at 121 (`' idx'` vs `' indices'`; ref margin 0.25, A2 gap **0.0000** — an exact tie), `flips_w` 0.0, `max|Δ|` 0.12061; cand diverges earlier (114 < 121) **and** `0.31493 > max(0.12061, 0.05)` ⇒ `FAIL`.

All eight rows (cand-night vs A), derived from the same call:

| tag | first_div | shared prefix | flips / flips_w | max\|Δlogp\| | mean\|Δlogp\| (derived) | margin at div (ref top1−top2) |
|---|---|---|---|---|---|---|
| prose | 31 | 31 | 0 / 0.0000 | 0.21797 | 0.02637 | 1.0000 |
| code | 114 | 114 | 0 / 0.0000 | 0.31493 | 0.01910 | 0.1250 |
| math | 27 | 27 | 0 / 0.0000 | 0.29664 | 0.04125 | 0.0000 (exact tie) |
| json | none | 18 | 0 / 0.0000 | 0.06564 | 0.00365 | – |
| shell | 118 | 118 | 0 / 0.0000 | 0.45515 | 0.02057 | 0.1250 |
| tool | none | 21 | 0 / 0.0000 | 0.33324 | 0.01587 | – |
| long | 3 | 3 | 0 / 0.0000 | 0.09890 | 0.03303 | 0.5000 |
| story | 1 | 1 | 0 / 0.0000 | 0.05129 | 0.05129 | 1.3750 |

`identical 2/8` (json, tool). Note `story`: `len_ref = 157` but `len_cand = 182`; shared_prefix = 1 is where it stops anyway.
Mean |Δ| is ≤ 0.05 on 7/8 rows and 0.05129 on story (one position); the **max** column is what fails `grade`.
The max |Δ| for `json` and `tool` is attained at position 2 — early, inside the prompt-conditioned region.

**Rounding and ties.** `top1_flips_weighted` → 4 dp, `max_abs_dlogprob` → 5 dp, no rounding on margins before the sum.
Ties: ref top-1/top-2 exactly equal at 18 of 1,476 positions in A (20 of 1,444 in A2); `sorted(..., key=-logprob)` is
stable, so a tie keeps the JSONL list order, and the candidate's side uses `max(...)`, which returns the first maximum.
A tie therefore has margin 0.0 (contributes 0 to `flips_w`) and can still produce a token divergence (math @27 above:
ref margin 0.0000, cand picked `'text'`).

## 3. What A3 would say if it were computed as approved (derived, not in any receipt)

- mean |Δlogprob| over the shared prefix, per row (table above): ≤ 0.05 on 7 of 8; story 0.05129 (n = 1).
- pooled over all 8 prefixes (sum of |Δ| ÷ total prefix positions) — compute with the snippet in the test; the number is
  not quoted here because it would be a *ninth* statistic nobody has gated on. Choose pooled vs per-prompt-worst before the next run (§5).
- top-1 clause: undefined until §5 exists.

## 4. Denominator and length caveats (as coded)

- `identical k/8`: denominator = fixture rows in `parity-A.jsonl` (8). A prompt counts identical when no token differs over
  the *shorter* of the two sequences. A candidate that stops early (EOS earlier) or runs longer with an identical shared
  part counts as identical; `len_ref/len_cand` are printed in the JSON only (`per_prompt`).
- Positions per prompt are fixture-length dependent (18…256), so any pooled position count weights long prompts
  ~14× json; `identical k/8` weights them equally. Neither is a rate per decoded token.
- The reference is vLLM Pg61 itself: A vs A2 (same engine, one run each) already differs on `code` @121 — the noise floor
  of the instrument is one divergence in eight and a 0.12 max |Δ|.

## 5. Planned: the teacher-forced fixed-prefix arm (SPEC — not implemented)

Purpose: measure top-1 agreement and logprob drift at **every** position under **identical** context, so a flip is
observable and the free-running first-divergence (which truncates the comparison and makes later positions incomparable)
stops hiding it.

Protocol (per fixture prompt, `w0/results/phase1/parity-A.jsonl`):
1. Feed our runtime the prompt, then Pg61-A's own emitted tokens `t_0 … t_{N-1}` as the forced prefix (the machinery exists:
   `first_light.step_cell_rows` / `gap_ledger ours-tf`, which already compare our argmax at each position with the forced
   next token; mapping Pg61 token strings → ids is `gap_ledger.load_targets`, gated `targets ok 8/8`).
2. At each position `i` capture our fp32 logits → our top-1 token, our log-softmax, our top-20 (fp32, no bf16 rounding on our side).
3. Per position, against Pg61's `top_logprobs[i]` (k = 20): record
   `agree_i` = (our top-1 == `t_i`); `ref_gap_i` = Pg61 top-1 logprob − Pg61 logprob of **our** top-1 token (`+∞` flagged `>top20` if not in its 20);
   `our_gap_i` = our top-1 logprob − our logprob of `t_i`; `dlp_i = |our logprob(t_i) − Pg61 logprob(t_i)|`.
4. Report (raw counts first, then ratios): `N` = Σ positions (the denominator: 1,476 for A at max_new 256 — equals the sum of `len_ref`, **not** truncated to multiples of 6 as the 6-row ledger cell is);
   `D` = #disagreements; `D_tie` = #disagreements at positions where Pg61's own top-1/top-2 gap is 0 (excluded from the strict count, reported);
   `D_w = Σ ref_gap_i` over disagreements (the margin-weighted flips A3 meant, now non-vacuous) and `Σ our_gap_i`;
   `mean dlp`, `max dlp` (both, pooled and worst-prompt), the per-prompt first disagreement, and the worst-5 positions with token strings.
5. Tie handling: a position where Pg61's top-1 and top-2 logprobs are equal (18 in A) agrees if our top-1 ∈ {tied tokens}; otherwise disagrees *and* is counted in `D_tie`. Pg61 logprobs are bf16-coarse, so "tie" means exactly equal values, not within ε; also report `near-tie` (gap ≤ 0.125).
6. Pre-register before the run (in the HANDOFF, not after): the pass bar (e.g. `D_w ≤ x` and `D ≤ y%`), whether it is pooled or worst-prompt, and the noise-floor comparison — the same arm applied to Pg61-A2 and `parity-pg61-nvfp4.jsonl` tokens as forced prefixes is **not** possible without the engine's logits, so the floor stays "vLLM vs vLLM" from the free-running table (A2 vs A, §2) until a vLLM teacher-forced run exists.
7. Determinism and provenance: self-replay bit-identical across two runs (A3 clause 1), eager == graph logits, the `receipt` block stamped (commit, dirty sha, switches, versions, clocks) — `bench/receipt.py`.

A first look (**measured**, an existing receipt of a different instrument — not this arm): `docs/gap-tf-static-20261107.json`,
6-row teacher-forced on Pg61's tokens, `argmax_agree_with_pg61` / `argmax_compared` per prompt: prose 233/252, code 249/252,
math 246/252, json 12/12, shell 248/252, tool 18/18, long 243/252, story 142/156 ⇒ **1,391 / 1,446 agree; 55 disagreements (3.8 %)**
— where the free-running table shows `flips = 0` on every prompt. That cell truncates each prompt to a multiple of 6 positions,
includes ties as disagreements, and says nothing about margins, so it is a floor on what the spec'd arm would report, not the arm.
