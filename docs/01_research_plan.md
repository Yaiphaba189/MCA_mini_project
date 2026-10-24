# Research Plan: Validated Commit Gates for Continually Adapted Language Models

Created: 2026-10-20 · Status: planning (no code yet)

## 0. One-paragraph summary

Continual-adaptation methods decide *when* to update (trigger) and *how* (adapter, replay, growth). Few decide *whether to keep* a candidate update: existing gated-update work (for example STABLE, arXiv:2510.16089) checks the stability of LoRA merges against a budget, but does not score candidates on held-out new-domain gain together with retention, and does not measure how well such a gate predicts the true value of an update. We build a **commit gate** for LoRA-style adapters: before merging a candidate adapter, score it on (a) held-out new-domain data and (b) a retention probe set, and reject or roll back predicted-harmful updates. We test whether this beats entropy-triggered adapter growth (the author's prior work) and always-update, at similar compute.

**Core claim to test (H1):** a gate that scores candidates on held-out new data plus a retention probe set yields better new-domain performance and less forgetting than trigger-only updating, at comparable compute.

## 1. Scope decisions (fix these early, change only with a reason)

| Decision | Choice | Why |
|---|---|---|
| Setting | LLM + LoRA adapters, text streams | Cheap update/hold/rollback; avoids RL compute |
| Narrow question | Commit decision only (RQ1). Not attribution, not locus-of-update | Scooping risk: one narrow question, move fast |
| Framing | "Validated commit gate", not "human-like world model" | World-model/cognitive framing is optional motivation; do not over-claim |
| Prior work reuse | Entropy-gated adapter growth = the trigger + the main baseline | Gate sits on top of it |

Out of scope: robotics/Dreamer experiments, localization of changed modules, in-context/rule-based update loci (list as future work).

## 2. Formalization

Stream of domain chunks D_1, D_2, ... (some recurring). Base weights θ. A trigger fires at time t and produces a candidate update Δθ (new or fine-tuned adapter on the recent buffer).

**Ground-truth update value (oracle, used for evaluation only):**
ΔR = R(θ + Δθ) − R(θ), measured on a large future/test slice of the new domain **and** on the full retention suite. Report two components: ΔNew and ΔOld (negative = forgetting).

**Gate (pre-commit, uses only information available at decision time):**
Score(Δθ) = [L_new(D_holdout; θ) − L_new(D_holdout; θ+Δθ)] − λ · [L_old(D_probe; θ+Δθ) − L_old(D_probe; θ)]

Commit iff Score > τ (and optionally ΔOld-probe ≤ ε as a hard constraint). Note the sign convention: a positive first term is improvement. (The pasted draft's formula subtracts losses in the wrong direction; fix this in the paper.)

**Key research content beyond the formula** (otherwise it is only an engineering check):
1. **Predictive validity:** how well does Score predict oracle ΔR? (AUROC for "harmful update", rank correlation with ΔR.)
2. **Probe-size and noise:** how small can hold-out and probe sets be before the gate becomes unreliable? (Compute/accuracy tradeoff.)
3. **Probe overfitting / leakage:** repeated gating against the same probe set adapts to it. Handle with rotating probe subsets plus a never-touched final retention suite.
4. **Calibrated threshold:** choose τ and λ on a separate validation stream (not the test stream); report sensitivity.
5. **Cheaper predictors (stretch):** a learned predictor of ΔR from cheap features (gradient alignment with probe gradients, hold-out loss delta, update norm, adapter-merge interference) vs. full probe evaluation.

## 3. Experimental design

### 3.1 Models
- Dev: small model (about 0.5–1.5B params, e.g. Qwen2.5-1.5B or Llama-3.2-1B) for fast iteration, many seeds.
- Confirmation: one 7–8B model (Llama-3-8B or Mistral-7B) with QLoRA, fewer seeds.
- Pick exact checkpoints at implementation time; record versions and licenses.

### 3.2 Streams (the shift types the survey says are under-tested)
| Stream | Description | Tests |
|---|---|---|
| S1 Abrupt domain shift | A → B (e.g. news → biomedical/legal/code) | basic adaptation vs forgetting |
| S2 Gradual drift | mixture ratio moves A→B over time | trigger-timing vs gate |
| S3 Recurring | A → B → A → B | does retention protect return to A |
| S4 Noise-only | high-entropy but non-informative text (shuffled/garbled/off-domain junk) | false-trigger/harmful-update rejection |
| S5 Compound / low-resource | several low-resource languages/domains (links to prior work) | where the author's entropy growth was built |

Candidate data (confirm licenses and availability at implementation): domain corpora from public sets (e.g. PubMed abstracts, legal text, code, news, Wikipedia in low-resource languages). Task-level alternative: sequences of classification/QA tasks. Pick one primary, one secondary.

### 3.3 Probes
- **Hold-out new probe:** most recent k examples from the shifting domain, withheld from candidate training. Vary k ∈ {64, 128, 256, 512}.
- **Retention probe:** sample from each past domain, stored as a small replay-free eval set. Vary size.
- **Final retention suite:** large, separate, never used by the gate. This is what forgetting is reported on.

### 3.4 Methods compared
| ID | Method |
|---|---|
| B0 | Static (no adaptation) |
| B1 | Always-update (fine-tune one shared LoRA at every trigger/chunk) |
| B2 | Fixed-schedule update |
| B3 | Entropy-triggered adapter growth (author's prior method) |
| B4 | Standard continual-LoRA baseline (e.g. replay LoRA; optionally O-LoRA/SD-LoRA/C-LoRA if reproducible) |
| B5 | Loss-spike trigger + rollback on hold-out only (ablation: no retention term) |
| **M1** | **B3 + commit gate (full: new + retention)** |
| M2 | Gate with retention only / new only (ablations) |
| M3 | Stretch: learned cheap predictor of ΔR instead of full probes |

### 3.5 Metrics
- New-domain performance: perplexity (and task accuracy if task stream).
- Forgetting: drop on final retention suite; backward transfer; average over the stream end.
- Recovery on recurrence (S3).
- **Harmful-update rate** (oracle ΔR < 0 or ΔOld below tolerance) among *committed* updates; **harmful-update rejection rate** (TPR) and **good-update false-reject rate** (FPR); gate AUROC.
- Compute: gate cost (forward passes on probes) plus training FLOPs/GPU-hours per committed update; also adapter count/parameters for growth methods.
- Stats: ≥5 seeds (dev model), bootstrap 95% CIs, paired comparisons across seeds; report distributions of per-update ΔR, not only means.

### 3.6 Ablations and analyses
1. λ and τ sweeps; sensitivity heatmap.
2. Probe sizes (cost vs. reliability curve).
3. Gate with noisy/misspecified probes (probe from wrong domain).
4. Which stream types the gate helps and where it hurts (report negative results honestly; a gate may reject too much on S2).
5. Gate on top of different triggers (entropy, loss spike, fixed schedule) to show it is trigger-agnostic.
6. Oracle gate (uses ground truth ΔR) as an upper bound.

## 4. Success criteria and kill criteria

- **Go:** M1 significantly reduces forgetting (CI excludes zero) without losing new-domain performance vs B3 on ≥2 of S1/S3/S4, at ≤1.3× compute; gate AUROC clearly above chance (target > 0.75).
- **Pivot:** if oracle gate gains are small (updates are rarely harmful in this setting), shift to streams where harm is real (noise, conflicting domains, tiny buffers) or reframe as an analysis paper of *when* updates harm.
- **Stop/reframe:** if the gate matches B3 everywhere, write it as a negative/analysis result for a workshop.

## 5. Work plan (about 12 weeks; adjust to compute)

| Weeks | Milestone | Deliverable |
|---|---|---|
| 1 | Verify literature: read in full fork ledger (2609.10954), CAWM, replay retention, SEGA, plus LoRA continual baselines; run citation search on fork ledger; re-check for scooping. Verify SPREAD or drop it. | `docs/literature_notes.md`, finalized gap matrix |
| 1–2 | Infrastructure: repo layout, config system, seeds, logging, data loaders, probe builders, evaluation harness | runnable skeleton, tests for eval |
| 2–3 | Reproduce B0, B1, B3 on S1 with the dev model | baseline table, sanity checks |
| 3–4 | Implement gate (M1) and oracle gate; measure ΔR for every candidate (fork: train candidate, evaluate both branches) | per-update ΔR dataset |
| 4–6 | Full streams S1–S5, all baselines, 5 seeds | main results |
| 6–8 | Ablations, sensitivity, probe-size study, trigger-agnostic test | ablation tables/figures |
| 8–9 | 7–8B confirmation run (reduced grid) | confirmation table |
| 9 | Stretch M3 (learned predictor) if time | optional |
| 10–11 | Writing, figures, appendix, reproducibility checklist | full draft |
| 12 | Internal review, fix, submit/arXiv | submission |

## 6. Risks

| Risk | Mitigation |
|---|---|
| Scooping (closest work is recent preprints) | Narrow claim, early arXiv/workshop posting once main result exists |
| Probe leakage / self-evaluation circularity | Separate final retention suite; rotate probes; report gate-vs-oracle gap |
| Gate overhead cancels any compute saving | Report compute honestly; test small probes; stretch M3 |
| Updates rarely harmful in chosen streams | S4 noise stream, conflicting-domain stream, tiny buffer; pivot rule above |
| Prior method details lost or not reproducible | Re-implement B3 from own notes first (week 2) and document differences |
| Over-claiming human-likeness | Keep cognitive framing in related work/discussion only, as hypothesis |
| Compute limits for 7–8B | Use QLoRA, short streams; keep 7–8B as confirmation only |
| Unverified citations | Every reference checked against the source before submission |

## 7. Reproducibility and project hygiene

- Fixed seeds, config files per run, results logged to disk (JSON/CSV), one command per experiment.
- Pin library versions; save adapter checkpoints only for committed updates plus a sample of rejected ones for analysis.
- Pre-register primary metrics and the go/kill criteria above before running the full grid.
- Keep a `docs/decisions.md` log for any deviation.

## 8. Immediate next steps (this week)

1. Read the six papers in full (list in section 5, week 1) and update the gap matrix.
2. Locate the existing entropy-gated adapter growth code and results; place it in this project or write down its exact specification.
3. Choose dev model and primary dataset/stream; check licenses.
4. Create the repo skeleton (`src/`, `configs/`, `experiments/`, `results/`, `docs/`).
5. Write the gate definition in `docs/decisions.md` with the corrected sign convention.
