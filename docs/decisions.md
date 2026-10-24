# Decisions log

Cite plan step numbers (for example `P0.5`). Append corrections below the pre-registration block; never edit it.

---

## Pre-registration block (P0.5) — dated 2026-10-24, written BEFORE any experiment run

**Status of values:** `ε_old` and the bound on harmful rate below are proposed defaults, to be confirmed by the author before the first run. Mark confirmation date here: `confirmed: ____`.

### Sign convention
Losses are mean token NLL (lower is better). Every gain is `before − after` (positive = good).

```
gain_new(Δθ) = L_new(D_holdout; θ) − L_new(D_holdout; θ+Δθ)
cost_old(Δθ) = L_old(D_probe;  θ+Δθ) − L_old(D_probe;  θ)
Score(Δθ)    = gain_new − λ · cost_old
Commit iff Score > τ   (optional hard constraint: cost_old ≤ ε)
Oracle: ΔNew = L_new(test; θ) − L_new(test; θ+Δθ);  ΔOld = L_old(suite; θ) − L_old(suite; θ+Δθ)
```

### Harmful update
`harmful := ΔNew ≤ 0  OR  ΔOld < −ε_old`, with **`ε_old = 0.02` nats/token** (proposed; roughly 2% perplexity increase on old domains).

### Primary metrics
1. New-domain test perplexity
2. Forgetting on the final retention suite (NLL_after − NLL_before per past domain)
3. Harmful-update rate among committed updates
4. Gate AUROC for `harmful` (plus TPR/FPR)
5. Compute: training tokens + gate forward tokens (oracle tokens reported separately and excluded)

### Go / pivot / stop (copied from `01_research_plan.md` §4)
- **Go:** M1 significantly reduces forgetting (CI excludes zero) without losing new-domain performance vs B3 on ≥2 of S1/S3/S4, at ≤1.3× compute; gate AUROC > 0.75.
- **Pivot:** oracle-gate gains small (updates rarely harmful) → streams where harm is real (noise, conflicting domains, tiny buffers) or reframe as analysis of *when* updates harm.
- **Stop/reframe:** M1 matches B3 everywhere → negative/analysis result for a workshop.

### Other
- Seeds: `[0, 1, 2, 3, 4]` (dev model). Seed changes data order, LoRA init, probe rotation; **not** the pool split.
- Harmful-rate bound used when choosing τ, λ on the validation stream (P6.7): ≤ 0.10 (proposed).
- Bootstrap: 10,000 resamples, 95% CI, paired across seeds.

---

## Compute (2026-10-24)
All model training and real-model evaluation on Google Colab (CUDA). Laptop: code editing and tiny-model CPU tests only. Results stored on Google Drive.

## Dev model (P0.3)
Default Qwen2.5-0.5B (`Qwen/Qwen2.5-0.5B`). Revision hash and license: **pending**, printed by `scripts/smoke_test.py` on Colab. Fall back/step up to Qwen2.5-1.5B only if 0.5B shows too little forgetting (P5.9).

## Datasets (P0.4)
**Pending.** Candidates in P3.1. License, split and download date to be recorded here.

## B3 specification (P0.6)
**Pending, needs author input.** The old entropy-triggered adapter-growth code/notes are not in this repo. Fill in (or point to the code to copy into `src/commitgate/legacy/`):
- entropy: which quantity, over which tokens (mean predictive entropy per token over the chunk?)
- threshold rule (fixed value, running mean + k·std, other?) and calibration
- when a new adapter is created (every trigger? capped count?)
- how adapters are combined at inference (sum of LoRA deltas, routing, merge?)
- rank / alpha / target modules (plan default: r=8, alpha=16, q_proj,v_proj)
- mark "re-implemented" if written from notes rather than copied

## Phase 1 finding (2026-10-24)
STABLE (arXiv:2510.16089) gates LoRA merges with a stability budget, so the plan's gap claim ("almost none decide whether to keep a candidate update") is too strong. Proposed narrower gap statement and differentiators are in `docs/literature_notes.md` (gap section). Awaiting author approval to change `01_research_plan.md` §0. SPREAD not found; dropped from related work (P1.4). B4 provisionally plain replay-LoRA (P1.2), pending confirmation.

## Approvals (2026-10-24, author said "do it" to the Phase 1 proposals)
- Gap statement in `01_research_plan.md` §0 reworded as proposed in `docs/literature_notes.md` (STABLE narrows the original claim). Main differentiator: predictive validity of the gate vs oracle ΔR (AUROC), plus the two-sided Score, probe-size/rotation study and trigger-agnostic test.
- B4 = plain replay-LoRA (P1.2). O-LoRA optional if time allows.
- Still open: titles for CAWM, SEGA, replay-retention paper; author's own full read of STABLE, fork ledger, SEA; manual "cited by" check for the fork ledger.
