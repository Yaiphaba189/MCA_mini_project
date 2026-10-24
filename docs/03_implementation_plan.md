# Implementation Plan: CommitGate

Created: 2026-10-20 · Companion to `01_research_plan.md` (what and why) and `02_paper_outline.md` (the paper). This file is the **how**: the order to build things, step by step, with a check at the end of each phase.

## How to use this document

- Work **one phase at a time, in order**. Do not start phase N+1 until every box in phase N's "Done when" list is ticked.
- Every phase has the same shape: **Goal → Prerequisites → Steps → Files → Done when → Pitfalls**.
- Steps are numbered `P<phase>.<step>` (for example `P3.4`). When you log a decision or a deviation in `docs/decisions.md`, cite the step number.
- Tick boxes in this file as you go. Do not delete finished phases.
- Time estimates follow the 12-week plan in `01_research_plan.md` §5. They assume part-time work on one Apple-silicon laptop for dev runs and rented GPU time for the 7–8B phase only.

**Compute decision (2026-10-24).** All model training and real-model evaluation runs on **Google Colab** (GPU), not on the local laptop. The laptop is used only for editing code and running tests with tiny/fake models on CPU. Use `notebooks/colab_run.ipynb` (clones the repo, installs, writes results to Google Drive). Wherever this plan says "MPS", read "CUDA on Colab" for real runs; keep MPS/CPU only for the tiny-model unit tests. Colab sessions reset, so runs must stay resumable (P7.3) and write to Drive.

## Phase map

| Phase | Name | Week | Output | Gate to next phase |
|---|---|---|---|---|
| 0 | Environment and decisions | 1 | working env, `decisions.md` | smoke test passes |
| 1 | Literature and gap check | 1 | `literature_notes.md` | gap matrix final |
| 2 | Skeleton and evaluation harness | 1–2 | package layout, eval, logging, tests | eval is deterministic and tested |
| 3 | Data, streams and probes | 2 | S1–S5 builders, probe builders | no leakage between splits (tested) |
| 4 | Adapter mechanics (candidate / commit / rollback) | 2–3 | train a candidate, snapshot, restore | restore is bit-exact |
| 5 | Baselines B0, B1, B2, B3 on S1 | 2–3 | baseline table | numbers sane vs. expectations |
| 6 | Oracle ΔR and the gate (M1) | 3–4 | per-update ΔR dataset, gate, AUROC | gate runs end to end on S1 |
| 7 | Full grid S1–S5, all methods, 5 seeds | 4–6 | main results | all runs complete, CIs computed |
| 8 | Ablations and analyses | 6–8 | ablation tables and figures | go / pivot decision made |
| 9 | 7–8B confirmation | 8–9 | confirmation table | reduced grid done |
| 10 | Stretch: learned predictor M3 | 9 | optional | skip freely |
| 11 | Figures, writing, release | 10–12 | manuscript, code release | pre-submission checklist ticked |

---

## Conventions (apply to every phase)

**Sign convention (fix once, never change).** Losses are mean token negative log-likelihood (NLL, lower is better). Every "gain" is `before − after`, so **positive = good**.

```
gain_new(Δθ) = L_new(D_holdout; θ) − L_new(D_holdout; θ+Δθ)        # > 0 means the update helped the new domain
cost_old(Δθ) = L_old(D_probe;  θ+Δθ) − L_old(D_probe;  θ)          # > 0 means forgetting
Score(Δθ)    = gain_new − λ · cost_old
Commit iff Score > τ   (optional hard constraint: cost_old ≤ ε)
```

Oracle quantities use the same signs but on the large test slice and the final retention suite:
`ΔNew = L_new(test; θ) − L_new(test; θ+Δθ)` (positive = good), `ΔOld = L_old(suite; θ) − L_old(suite; θ+Δθ)` (negative = forgetting).

**Harmful update (pre-register in P0.5):** `ΔNew ≤ 0` **or** `ΔOld < −ε_old`. Pick `ε_old` before running anything.

**Three evaluation sets, three jobs, never mixed:**

| Set | Used by | Used for |
|---|---|---|
| Hold-out new probe + rotating retention probe | the gate | commit decision (cheap, small) |
| Validation stream | you | choosing τ, λ, learning rates |
| Final retention suite + test slice | metrics only | reporting forgetting and oracle ΔR. **Never read by the gate or by tuning.** |

**Run discipline.** Every run = one YAML config + one seed + one output folder `results/<exp_name>/<method>/<stream>/seed<k>/`. Every run writes `config.yaml` (resolved), `log.jsonl` (one line per event), and `summary.json`. One command per experiment: `python scripts/run_experiment.py --config configs/<file>.yaml --seed <k>`.

**Naming.** Methods use the IDs from the research plan (B0–B5, M1–M3). Streams are `s1_abrupt`, `s2_gradual`, `s3_recurring`, `s4_noise`, `s5_lowres`.

---

## Phase 0 — Environment and decisions (week 1)

**Goal.** A reproducible environment and every early decision written down.

**Prerequisites.** The existing `.venv` (Python 3.11, torch 2.14, transformers 5.x, datasets). Apple-silicon MPS is available; CUDA is not (relevant for Phase 9).

**Steps**

- [ ] **P0.1** Activate the env: `source .venv/bin/activate`. Confirm `python -c "import torch; print(torch.backends.mps.is_available())"` prints `True`.
- [ ] **P0.2** Install the missing packages and add them to `pyproject.toml`:
  - core: `peft`, `accelerate`, `pyyaml`, `scipy`, `scikit-learn`, `pandas`, `tqdm`
  - extras: keep `plot = ["matplotlib"]`; add `cuda = ["bitsandbytes"]` (only installed on the rented GPU box in Phase 9)
  - then `pip install -e ".[dev,plot]"` and freeze versions to `requirements.lock` (`pip freeze > requirements.lock`).
- [ ] **P0.3** Choose the dev model. Default: **Qwen2.5-0.5B** for the first runs (fits MPS comfortably), step up to **Qwen2.5-1.5B** only if 0.5B shows too little forgetting to measure. Record exact checkpoint name, revision hash and license in `docs/decisions.md`.
- [ ] **P0.4** Choose the primary stream pair and the data (see P3.1). Record dataset names, splits, license and download date. Primary = perplexity on domain text. Secondary task stream is optional and only added after Phase 7.
- [ ] **P0.5** Create `docs/decisions.md` and write the **pre-registration block** (date it, do not edit afterwards; append corrections below it):
  - the corrected gate formula and sign convention above
  - the harmful-update definition and the value of `ε_old`
  - primary metrics (new-domain test perplexity, forgetting on the final retention suite, harmful-update rate, gate AUROC, compute)
  - go / pivot / stop criteria copied from `01_research_plan.md` §4
  - seed list: `[0, 1, 2, 3, 4]`
- [ ] **P0.6** Write the **exact specification of B3** (the entropy-triggered adapter growth from your prior work): what entropy is computed, over which tokens, the threshold rule, when a new adapter is created, how adapters are combined at inference, and rank/alpha/target modules. If you still have the old code, copy it into `src/commitgate/legacy/` and note the differences. If not, write the spec from notes and mark it "re-implemented".
- [ ] **P0.7** Smoke test: load the dev model, run one forward pass on MPS, print the loss on one sentence. Save the snippet as `scripts/smoke_test.py`.

**Files.** `pyproject.toml`, `requirements.lock`, `docs/decisions.md`, `scripts/smoke_test.py`.

**Done when**
- [ ] `python scripts/smoke_test.py` prints a finite loss on MPS
- [ ] `docs/decisions.md` contains the dated pre-registration block and the B3 specification
- [ ] model checkpoint, dataset names and licenses are recorded

**Pitfalls.** Use `float32` or `bfloat16` on MPS; avoid `float16` training (NaNs are common). Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if an op is missing. Do not pick a dataset you cannot redistribute or cite.

---

## Phase 1 — Literature and gap check (week 1, runs in parallel with Phase 0)

**Goal.** Make sure the claim is still novel and the related-work table is verified.

**Steps**

- [ ] **P1.1** Read in full: fork ledger (2609.10954), CAWM, the replay-retention paper, SEGA. Take notes per paper: setting, what is gated or measured, what they do *not* do.
- [ ] **P1.2** Read two continual-LoRA baselines you will compare against (pick from O-LoRA, SD-LoRA, C-LoRA, or plain replay-LoRA). Decide which one is reproducible for B4.
- [ ] **P1.3** Run a citation search on the fork-ledger paper (papers that cite it) and a keyword search for "commit gate", "update validation", "rollback", "continual adapter". Record the **queries and the date**.
- [ ] **P1.4** Verify SPREAD exists and says what you think; otherwise drop it from related work.
- [ ] **P1.5** Fill the coverage matrix (rows = papers, columns = trigger / predicts update value / commit-reject / retention check / LoRA). Save as `docs/literature_notes.md`.

**Done when**
- [ ] `docs/literature_notes.md` exists with per-paper notes and the matrix
- [ ] every paper you plan to cite was actually read (mark the rest "unverified")
- [ ] the gap statement in `01_research_plan.md` §0 still holds, or the deviation is in `decisions.md`

**Pitfalls.** Do not cite from memory. Re-run P1.3 again two weeks before submission (Phase 11).

---

## Phase 2 — Skeleton and evaluation harness (weeks 1–2)

**Goal.** A package whose evaluation function is deterministic and tested, before any training exists.

**Target layout** (create empty modules with docstrings first, then fill):

```
src/commitgate/
  config.py            # dataclasses + YAML loading, config merging, resolved-config dump
  seed.py              # set_seed(seed): python, numpy, torch (+ mps)
  io_utils.py          # JSONL logger, summary writer, run-dir creation
  data/
    sources.py         # raw dataset loaders (HF datasets), license metadata
    streams.py         # S1–S5 builders -> Stream object
    probes.py          # hold-out, retention probe, final suite, rotation
    tokenize.py        # tokenize + chunk to fixed length
  model/
    load.py            # load base model + tokenizer, device/dtype
    adapters.py        # LoRA create / train / snapshot / restore / combine
    train.py           # fit_candidate(buffer) -> candidate
  eval/
    nll.py             # mean token NLL / perplexity on a tokenized set
  triggers/
    entropy.py         # B3 trigger
    loss_spike.py      # B5 trigger
    fixed.py           # B2 trigger
  gate.py              # Score, decision rule, rotating probe selection
  oracle.py            # ΔNew / ΔOld on test slice + final suite
  methods/             # b0_static.py, b1_always.py, b2_fixed.py, b3_entropy.py, ...
  runner.py            # the stream loop (see Phase 5)
  metrics.py           # forgetting, BWT, harmful rate, TPR/FPR, AUROC, compute
  stats.py             # bootstrap CIs, paired comparison
configs/               # one YAML per experiment
scripts/               # run_experiment.py, aggregate.py, make_figures.py, smoke_test.py
tests/                 # mirrors src layout
results/               # git-ignored run outputs
```

**Steps**

- [x] **P2.1** Create the folders and empty modules above. Add `configs/base.yaml` with: model name, device, dtype, seed, sequence length (start 256), batch size, LoRA (`r=8`, `alpha=16`, `dropout=0.05`, target `q_proj,v_proj`), learning rate (`2e-4`), steps per candidate, probe sizes, `lambda`, `tau`, `eps`.
- [x] **P2.2** Implement `seed.py` and `config.py` (load YAML → dataclass, allow `--set key=value` overrides, write the resolved config into the run folder).
- [x] **P2.3** Implement `io_utils.py`: `RunLogger.log(event, **fields)` appends a JSON line with timestamp; `write_summary(dict)`.
- [x] **P2.4** Implement `eval/nll.py`: `nll(model, token_batches) -> float` (mean over non-padding tokens, `torch.no_grad`, `model.eval()`, restores train mode afterwards). Also return the token count so costs can be logged.
- [x] **P2.5** Write tests in `tests/test_eval.py`:
  - same model + same data → identical NLL twice (determinism)
  - padding does not change the result
  - NLL of a random-init tiny model is close to `log(vocab_size)`
  - perplexity = `exp(nll)`
- [x] **P2.6** Add a `compute counter` utility: counts forward tokens and training tokens per run (used for the compute metric in Phase 7). Unit-test it.
- [x] **P2.7** Add `.gitignore` (`results/`, `.venv/`, `*.pt`, `__pycache__`) and `git init` the project so every result can be tied to a commit hash. Log the commit hash in each run's `summary.json`.

**Done when**
- [x] `pytest -q` passes
- [x] a run folder is created with `config.yaml` and `log.jsonl` by a dummy script
- [x] NLL on a fixed sentence is identical across two processes

**Pitfalls.** Mean over *tokens*, not over sequences, and use the same definition everywhere. Mask padding in both the loss and the token count.

---

## Phase 3 — Data, streams and probes (week 2)

**Goal.** Turn raw corpora into streams with strictly separated splits.

**Steps**

- [ ] **P3.1** Pick sources (confirm licenses; record in `decisions.md`). Starting suggestion:
  - Domain A (general/news): Wikipedia or a news corpus subset
  - Domain B (biomedical): PubMed abstracts subset
  - Domain C (legal) and D (code): one public set each, used for S3 variants and the retention suite
  - Noise (S4): shuffled-token text, garbled characters, and off-domain junk (generate programmatically from domain A so the entropy is high but information is absent)
  - Low-resource (S5): two or three low-resource-language Wikipedias
- [ ] **P3.2** `data/sources.py`: one loader per source returning a list of raw text strings, with a fixed sampling seed and a cached local copy under `data/cache/` (git-ignored).
- [ ] **P3.3** `data/tokenize.py`: tokenize, concatenate, split into fixed-length blocks (`seq_len` from config). Cache tokenized arrays as `.npy`.
- [ ] **P3.4** Define the `Stream` object: a list of **chunks**; each chunk = `(domain_id, tokens[N, seq_len])`. Chunk size default 256 blocks; stream length default 40 chunks. Builders:
  - `s1_abrupt`: 20 × A, then 20 × B
  - `s2_gradual`: mixture ratio moves linearly from 100% A to 100% B over 40 chunks
  - `s3_recurring`: A, B, A, B in blocks of 10 chunks
  - `s4_noise`: A chunks with noise chunks inserted at fixed positions (for example chunks 10–14 and 25–29)
  - `s5_lowres`: sequence of 3 low-resource languages, 13–14 chunks each
- [ ] **P3.5** `data/probes.py`, for each domain keep **three disjoint pools** carved out once, with a fixed seed, before any stream is built:
  1. `stream pool` (feeds the stream)
  2. `probe pool` (source for hold-out new probe and retention probe)
  3. `final pool` (final retention suite + test slice; the gate may never touch it)
  Provide `holdout_new(chunk_history, k)` (the most recent `k` blocks withheld from candidate training), `retention_probe(domains_seen, size, round)` (rotating subset; round index picks a different subset each time), and `final_suite(domains_seen)`.
- [ ] **P3.6** Make the probe sizes configurable: `k ∈ {64, 128, 256, 512}` for the hold-out, `{32, 64, 128}` per old domain for the retention probe. Defaults: 128 and 64.
- [ ] **P3.7** Tests in `tests/test_data.py`:
  - no block appears in two of {stream pool, probe pool, final pool}
  - hold-out blocks are never in the candidate's training buffer
  - same seed → identical stream
  - rotation: consecutive rounds give different but overlapping-bounded subsets
  - S4 noise chunks have higher unigram entropy than clean chunks

**Done when**
- [ ] `python -m commitgate.data.streams --stream s1_abrupt --summary` prints chunk/domain layout
- [ ] leakage tests pass
- [ ] token counts per pool are recorded in `decisions.md`

**Pitfalls.** Deduplicate near-identical documents before splitting (PubMed and Wikipedia contain repeats). Carve the final pool *first* so you cannot accidentally leak it later.

---

## Phase 4 — Adapter mechanics: candidate, commit, rollback (weeks 2–3)

**Goal.** A candidate update is a first-class object that can be trained, scored, committed, or discarded with zero side effects.

**Steps**

- [ ] **P4.1** `model/load.py`: `load_base(cfg)` returns `(model, tokenizer)` on the chosen device with the base weights frozen.
- [ ] **P4.2** Spike test (30 minutes, script only): with `peft`, can you (a) add a second LoRA adapter, (b) activate several adapters at once, (c) run a backward pass only on the new one? Behaviour differs by `peft` version, so check the installed version. Record the result in `decisions.md`. **If any answer is no**, write a minimal own `LoRALinear` wrapper (about 60 lines) and use it instead of peft for the dev phase; the interface below stays the same.
- [ ] **P4.3** `model/adapters.py` defines `AdapterState` (dict of tensors for one LoRA adapter) and these functions, whose signatures must stay stable:
  - `new_candidate(model, mode)`: `mode="fresh"` creates a new adapter (growth, used by B3/M1); `mode="finetune"` copies the current shared adapter (used by B1).
  - `snapshot(model) -> Snapshot` and `restore(model, snapshot)` (restores adapter weights **and** optimizer state if you keep one).
  - `commit(model, candidate)` and `discard(model, candidate)`.
  - `active_adapters(model)`: the list that contributes at inference. Use the combination rule from the B3 spec (P0.6).
  - `adapter_param_count(model)` for the compute table.
- [ ] **P4.4** `model/train.py`: `fit_candidate(model, candidate, train_tokens, cfg) -> TrainStats`. Plain AdamW, fixed number of steps (config), gradient clipping 1.0, fixed batch size, cosine or constant LR (pick one, keep it). Returns final train loss and the number of training tokens processed (for the compute counter).
- [ ] **P4.5** Candidate lifecycle helper used by every method:

```
snap = snapshot(model)
cand = new_candidate(model, mode)
fit_candidate(model, cand, buffer_train_tokens, cfg)
# ... gate or oracle evaluates model-with-candidate here ...
if keep: commit(model, cand)
else:    restore(model, snap)          # must leave the model exactly as before
```

- [ ] **P4.6** Tests in `tests/test_adapters.py`:
  - after `restore`, NLL on a fixed batch is **bit-identical** to before the candidate was trained
  - training a candidate changes only that candidate's parameters; base weights and earlier adapters are unchanged (compare checksums)
  - `commit` makes the candidate contribute to `active_adapters`; `discard` does not
  - same seed → same candidate weights after `fit_candidate`

**Done when**
- [ ] all adapter tests pass on MPS (and on CPU for the exactness test)
- [ ] a script trains one candidate on domain B and shows B-NLL fall and A-NLL rise, then restores and shows both return exactly

**Pitfalls.** Dropout must be off in evaluation. Do not let the optimizer keep momentum from a discarded candidate. Clear MPS cache between candidates if memory climbs (`torch.mps.empty_cache()`).

---

## Phase 5 — Stream loop and baselines B0–B3 on S1 (weeks 2–3)

**Goal.** One runner that every method plugs into, and trustworthy baseline numbers on S1.

**Steps**

- [ ] **P5.1** Define the method interface in `methods/base.py`:

```python
class Method:
    def on_chunk(self, ctx) -> Decision   # ctx: model, chunk, buffer, history, probes, counters, logger
    # Decision = {"trigger": bool, "candidate_mode": "fresh"|"finetune"|None, "commit": bool, "info": {...}}
```

- [ ] **P5.2** `runner.py`: the loop. For each chunk `t`:
  1. (optional) evaluate the current model on the cheap tracking set and log it,
  2. append the chunk to the buffer,
  3. ask the method for a decision,
  4. if triggered, run the candidate lifecycle from P4.5,
  5. every `eval_every` chunks (default 5) evaluate on a **tracking** slice, and at the very end evaluate on the **final suite**,
  6. log every event to `log.jsonl`, update compute counters.
- [ ] **P5.3** Implement **B0 static** (never triggers). Evaluate A and B NLL. This fixes the "before any adaptation" reference numbers.
- [ ] **P5.4** Implement **B1 always-update**: trigger on every chunk, `finetune` one shared adapter, always commit.
- [ ] **P5.5** Implement **B2 fixed-schedule**: trigger every `m` chunks (config, default 5), `finetune`, always commit.
- [ ] **P5.6** Implement `triggers/entropy.py` and **B3**: compute the predictive entropy on the chunk with the current model as specified in P0.6; trigger when it exceeds the threshold; `fresh` adapter; always commit. Calibrate the entropy threshold on the **validation stream** (build a separate validation stream with a different seed) and write the value to `decisions.md`.
- [ ] **P5.7** `metrics.py` first version: final-suite NLL per domain, forgetting = `NLL_after − NLL_before` on each past domain, average NLL over the stream, number of committed updates, adapter count, token counters.
- [ ] **P5.8** Run B0–B3 on `s1_abrupt` with **seed 0** and the dev model. Put results in `results/baselines_s1/` and a markdown table in `docs/baseline_table.md`.
- [ ] **P5.9** Sanity checks (all must hold, otherwise debug before moving on):
  - B1 lowers domain-B perplexity vs B0
  - B1 raises domain-A perplexity vs B0 (visible forgetting). If it does not, forgetting is too small to measure; raise LR/steps, shorten the buffer, or move to the 1.5B model
  - B3 triggers at the A→B boundary and rarely inside A
  - compute counters agree with a hand calculation for one run

**Done when**
- [ ] `python scripts/run_experiment.py --config configs/s1_baselines.yaml --seed 0` produces all four method folders
- [ ] baseline table written and sanity checks ticked
- [ ] the runner, not the methods, owns evaluation (so methods cannot cheat)

**Pitfalls.** Keep the training-token budget per update identical across methods unless the method specifies otherwise, and state it. Do not tune any baseline on the final suite.

---

## Phase 6 — Oracle ΔR and the gate M1 (weeks 3–4)

**Goal.** For every candidate update, record both what the gate *predicted* and what *actually* happened.

**Steps**

- [ ] **P6.1** `oracle.py`: `oracle_delta(model, candidate_ctx) -> {dNew, dOld_per_domain, dOld}` evaluates before/after on the **test slice** and the **final retention suite**. Run it for every candidate, including rejected ones, but flag it `oracle_only=True` so it never feeds a decision. Log its cost separately from the gate's cost.
- [ ] **P6.2** `gate.py`: implement
  - `gate_score(model, snap, cand, holdout, probe, lam) -> {score, gain_new, cost_old}`; evaluate the same probes with and without the candidate (both numbers logged)
  - `decide(score, cost_old, tau, eps) -> bool`
  - `rotating_probe(round)` using `probes.retention_probe`
- [ ] **P6.3** Tests in `tests/test_gate.py` using a tiny fake model: a candidate that is better on new and neutral on old → positive score; a candidate that improves new but damages old → score falls as `lam` rises; a noise candidate → score ≤ 0. Check the sign convention explicitly.
- [ ] **P6.4** `methods/m1_gate.py`: B3's trigger + `fresh` candidate + gate decision. On reject, `restore` and log the rejection reason. Add the *same* logic as a wrapper `GateOn(trigger)` so it can sit on top of any trigger (needed for the trigger-agnostic ablation in Phase 8).
- [ ] **P6.5** Implement the **oracle gate**: same as M1 but the decision uses `ΔNew`/`ΔOld` from the oracle. This is the upper bound.
- [ ] **P6.6** Build the **per-update dataset**: a loop that, for every candidate in every run, stores one row: `run_id, t, domain, trigger_value, gain_new, cost_old, score, dNew, dOld, harmful, committed, gate_tokens, train_tokens`. Write it to `results/<exp>/updates.parquet` (or CSV).
- [ ] **P6.7** Choose `tau` and `lambda` on the **validation stream** only: grid `lambda ∈ {0.5, 1, 2, 4}`, `tau ∈ {0, small positive values}`; pick the setting that maximizes validation `ΔNew + ΔOld` summed over committed updates subject to harmful rate ≤ a bound you pre-register. Write the chosen values and the grid result into `decisions.md`.
- [ ] **P6.8** Run B3, M1 and the oracle gate on `s1_abrupt` and `s4_noise` (seed 0). Compute gate **AUROC** against `harmful` and the Spearman correlation of `score` with `dNew + dOld`.

**Done when**
- [ ] gate tests pass; sign convention confirmed by test
- [ ] `updates.parquet` exists and is non-empty for both streams
- [ ] on S4, M1 rejects most noise-chunk candidates while B3 commits them (this is the first visible effect; if not, inspect gate scores before going further)
- [ ] chosen `tau`/`lambda` came from the validation stream (written down)

**Pitfalls.** Evaluate before/after with the *same* probe blocks and the same tokenization. Never let `dOld` (oracle) leak into the gate path; keep oracle code in a module the gate does not import. If AUROC is near 0.5, first check probe size and the retention probe's domain coverage before concluding the idea fails.

---

## Phase 7 — Full grid: S1–S5, all methods, 5 seeds (weeks 4–6)

**Goal.** The main results table.

**Steps**

- [ ] **P7.1** Implement the remaining methods: **B4** (your chosen continual-LoRA baseline from P1.2) and **B5** (loss-spike trigger + rollback on hold-out only). Add the M2 ablations as config flags on M1 (`use_retention`, `use_holdout`).
- [ ] **P7.2** Write `configs/grid.yaml` listing methods × streams × seeds. Estimate wall-clock first: time one S1 run, multiply by `methods × streams × seeds`, and decide whether to shorten streams (fewer chunks) or run overnight. Put the estimate in `decisions.md`.
- [ ] **P7.3** `scripts/run_grid.py`: loops the grid, **skips runs whose `summary.json` already exists** (so you can resume after a crash), and writes a `grid_status.csv` (run, status, seconds).
- [ ] **P7.4** Run the grid in the order: S1 → S4 → S3 → S2 → S5. Check results after each stream before launching the next (catch bugs early rather than after 40 runs).
- [ ] **P7.5** `scripts/aggregate.py`: gather every `summary.json` and `updates.parquet` into `results/<exp>/all_runs.csv` and `all_updates.csv`.
- [ ] **P7.6** `stats.py` and the main table: mean ± bootstrap 95% CI over seeds (10,000 resamples), **paired** differences M1 − B3 per seed, and the distribution of per-update `ΔR`.
- [ ] **P7.7** Compute table: per method, training tokens + gate forward tokens + oracle tokens reported separately; oracle cost is **excluded** from method cost (it is evaluation only), gate cost is **included**.
- [ ] **P7.8** Evaluate the go criterion from `01_research_plan.md` §4 and record the outcome (go / pivot / stop) in `decisions.md`, with the numbers.

**Done when**
- [ ] `grid_status.csv` shows every run complete
- [ ] main table with CIs exists as `docs/results_main.md`
- [ ] go / pivot / stop decision is written

**Pitfalls.** One seed must change the data order, the LoRA init and the probe rotation, but **not** the pool split (keep the split fixed so seeds are comparable). Never rerun a method with a tuned setting without giving every baseline the same tuning budget.

---

## Phase 8 — Ablations and analyses (weeks 6–8)

**Goal.** Explain *why* it works or where it fails.

**Steps** (each produces one table or figure; script names in brackets)

- [ ] **P8.1** `λ`/`τ` sensitivity heatmap on the validation stream and on test (`make_figures.py --fig sensitivity`).
- [ ] **P8.2** Probe-size study: hold-out `k` × retention size grid; plot AUROC and harmful-update rate against gate cost (`--fig probe_size`).
- [ ] **P8.3** Wrong-probe robustness: replace the retention probe with a different domain or with noise; measure degradation.
- [ ] **P8.4** Trigger-agnostic test: `GateOn(entropy)`, `GateOn(loss_spike)`, `GateOn(fixed)`; compare each to its ungated version.
- [ ] **P8.5** Component ablations M2: new-only, retention-only, both.
- [ ] **P8.6** Oracle gap: M1 vs oracle gate vs B3 on all streams; report which streams have headroom.
- [ ] **P8.7** Failure analysis: list S2 (gradual) updates the gate wrongly rejected; look at five by hand; write what you see.
- [ ] **P8.8** Probe overfitting check: run a long stream with a **fixed** probe vs a **rotating** probe and compare the gap between probe loss and final-suite loss over time.
- [ ] **P8.9** Re-evaluate the success criteria; if the oracle gate also shows small gains, switch to the pivot in `01_research_plan.md` §4 and log it.

**Done when**
- [ ] every ablation has a saved figure/table under `results/figures/` and an entry in `docs/results_ablations.md`
- [ ] negative findings are written down, not dropped

---

## Phase 9 — 7–8B confirmation (weeks 8–9)

**Goal.** Show the result is not an artifact of the small model, using a reduced grid.

**Steps**

- [ ] **P9.1** Rent one CUDA GPU (24 GB is enough for QLoRA on a 7–8B model). Set up the same repo, `pip install -e ".[cuda]"`, copy `requirements.lock`, log the GPU type.
- [ ] **P9.2** Add a `quantization: 4bit` option in `model/load.py`; confirm the adapter tests (P4.6) still pass on CUDA with the quantized base.
- [ ] **P9.3** Reduce the grid: methods {B0, B1, B3, M1, oracle gate} × streams {S1, S3, S4} × 3 seeds, shorter streams (state the length).
- [ ] **P9.4** Reuse `tau`/`lambda` from the dev model **as a first try** and also run one re-tuned setting on the validation stream; report both.
- [ ] **P9.5** Aggregate with the same scripts into `docs/results_confirmation.md`; mark clearly that this is a reduced grid.

**Done when**
- [ ] confirmation table exists and states what differs from the dev grid
- [ ] cost spent and GPU-hours are noted in `decisions.md`

**Pitfalls.** Save results to disk after every run and pull them locally often; rented machines disappear. Keep the 8B runs resumable (P7.3).

---

## Phase 10 — Stretch: learned cheap predictor M3 (week 9, optional)

**Goal.** Predict `ΔR` from cheap features without full probe evaluation. Skip if Phase 8 is behind.

**Steps**

- [ ] **P10.1** From `all_updates.csv`, build features per update: gradient cosine between candidate and probe gradients, hold-out loss delta, update norm, adapter-merge interference. (Add the extra feature logging to the candidate lifecycle; this needs a re-run of S1 with the new fields.)
- [ ] **P10.2** Train a small regressor/classifier (logistic regression or gradient boosting from scikit-learn) with **leave-one-stream-out** or leave-one-seed-out validation.
- [ ] **P10.3** Plug it in as gate M3, compare AUROC and compute against the full-probe gate.

**Done when**
- [ ] result table exists, or the phase is explicitly skipped in `decisions.md`

---

## Phase 11 — Figures, writing, release (weeks 10–12)

**Goal.** A defensible manuscript and a runnable release.

**Steps**

- [ ] **P11.1** Fix the figure set: Fig 1 pipeline diagram; main results table; forgetting curves; gate ROC; λ/τ heatmap; cost-vs-reliability curve; per-update ΔR distributions. One script call regenerates all: `python scripts/make_figures.py --all`.
- [ ] **P11.2** Write sections in the order given in `02_paper_outline.md` milestones: 3–5 after Phase 5–6 data exist, 6–7 after Phase 8, abstract/intro/limitations last.
- [ ] **P11.3** Re-run the gap search (P1.3) and update related work.
- [ ] **P11.4** Citation verification: every reference opened and checked against the source.
- [ ] **P11.5** Walk the pre-submission checklist in `02_paper_outline.md` and tick each item with evidence (file or table).
- [ ] **P11.6** Release prep: README with one-command reproduction for S1, license file, data-license table, config + seed list, `requirements.lock`.
- [ ] **P11.7** Reproduce S1 from a **fresh clone** in a new venv to confirm the release works.

**Done when**
- [ ] checklist in `02_paper_outline.md` fully ticked
- [ ] fresh-clone reproduction reproduces the S1 numbers within CI

---

## Cross-phase checklists

**Before starting any experiment run**
- [ ] working tree committed (commit hash will be logged)
- [ ] config in `configs/`, not hard-coded
- [ ] final retention suite untouched by the code path under test
- [ ] seed set; output folder does not already hold a different config

**If something looks wrong**

| Symptom | Check first |
|---|---|
| No forgetting visible with B1 | learning rate, steps per candidate, buffer size, model size (P5.9) |
| Gate AUROC about 0.5 | probe size, retention-probe domain coverage, sign convention, probe rotation |
| Gate rejects everything | `tau` too high, `lambda` too high, retention probe too noisy |
| Gate accepts everything | `tau` too low, hold-out overlaps the training buffer (leakage) |
| NLL differs between runs | dropout in eval mode, seed, MPS non-determinism (compare on CPU) |
| MPS out of memory | smaller batch, shorter `seq_len`, `torch.mps.empty_cache()`, smaller model |

**Risks that map to phases** (full list in `01_research_plan.md` §6): scooping → P1.3 and P11.3; probe leakage → P3.5, P3.7, P8.8; gate overhead → P7.7; updates rarely harmful → P6.8 and the pivot rule in P8.9; prior method not reproducible → P0.6; 7–8B compute → P9 kept as reduced confirmation.

## Status log

| Phase | Started | Finished | Notes |
|---|---|---|---|
| 0 | | | |
| 1 | 2026-10-24 | partial | Notes in `docs/literature_notes.md`. Only abstract/HTML-level reading; CAWM, SEGA, SPREAD, replay-retention paper not found. STABLE (2510.16089) narrows the gap. |
| 2 | 2026-10-24 | 2026-10-24 | Skeleton + eval harness done; 12 CPU tests pass |
| 3 | | | |
| 4 | | | |
| 5 | | | |
| 6 | | | |
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |
| 11 | | | |
