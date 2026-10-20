# Paper Outline: From Plan to Manuscript

Working title: **Check Before You Commit: Validated Update Gates for Continually Adapted Language Models**

Target: workshop first (e.g. a NeurIPS 2026-cycle continual-learning or continual-world-models style workshop, or CoLLAs), then extend to a conference (ICLR/NeurIPS) with the 7–8B results. Check deadlines and formats when the main results exist.

## One-sentence contribution
Trigger-based adapter updating can silently hurt; a pre-commit gate that scores a candidate on held-out new data and a retention probe prevents harmful merges at small cost.

## Contributions (claim only what the experiments support)
1. Formulation: separate *trigger* from *commit*; define per-update value ΔR (new and old components) for adapter updates.
2. A pre-commit gate usable on top of any trigger, plus analysis of how well its score predicts true update value.
3. Benchmark streams (abrupt, gradual, recurring, noise-only, low-resource) with harmful-update metrics.
4. Empirical study: forgetting, new-domain performance, compute, probe-size sensitivity, and when the gate fails.

## Section map

| Section | Content | Source in plan | Figures/tables |
|---|---|---|---|
| Abstract | Problem, gate, key numbers, honest scope | – | – |
| 1 Introduction | Trigger ≠ verdict; gap; contributions | plan §0–1 | Fig 1: trigger → candidate → gate → commit/rollback |
| 2 Related Work | Continual LoRA; continual TTA gating (CoTTA, SEGA, etc.); event-triggered learning; update-value measurement (fork ledger); continual world models; latent-cause theory as motivation only | survey §2–3 | Table: coverage matrix (trigger / predict value / commit) |
| 3 Problem Setup | Stream, candidate, ΔNew, ΔOld, harmful update definition | plan §2 | notation table |
| 4 Method | Gate score, thresholds, probe design, rotating probes, optional learned predictor | plan §2 | Algorithm 1 |
| 5 Experimental Setup | Models, streams, baselines, metrics, seeds | plan §3 | stream diagram, baseline table |
| 6 Results | Main comparison; harmful-update rate; recurrence; noise-only; compute | plan §3.5 | Main table with 95% CIs; forgetting curves; gate ROC |
| 7 Analysis | λ/τ sensitivity; probe size; trigger-agnostic; oracle gap; failure cases | plan §3.6 | Heatmaps; cost-vs-reliability curve |
| 8 Discussion & Limitations | Probe availability assumption; scale; domain coverage; cognitive framing as hypothesis only | plan §6 | – |
| 9 Conclusion | Summary, future work (attribution, locus of update, control tasks) | – | – |
| Appendix | Hyperparameters, data licenses, extra seeds, per-update ΔR distributions, reproducibility checklist | plan §7 | – |

## Writing rules
- Every claim ties to a table/figure; negative results stay in.
- Cite only papers actually read; mark and re-verify any cited from memory. Drop or verify SPREAD.
- Do not claim "first" unless a documented search (with date and queries) supports it; prefer "to our knowledge, we found no work that…".
- Do not claim human-brain equivalence.
- Report compute fairly, including gate cost.

## Milestones to paper
1. Main result on S1 + S3 with dev model → write Sections 3–5 immediately.
2. All streams + ablations → write Sections 6–7.
3. 7–8B confirmation → finalize abstract, intro, limitations.
4. Pre-submission checklist: seeds and CIs, license check, code release plan, citation verification, anonymization, venue format.

## Pre-submission checklist
- [ ] Gap re-checked within 2 weeks of submission (new arXiv search)
- [ ] All baselines tuned with equal budget
- [ ] Final retention suite never used for gating or tuning
- [ ] ≥5 seeds, bootstrap CIs
- [ ] Gate overhead included in compute comparison
- [ ] Limitations section written
- [ ] Code/config release prepared
