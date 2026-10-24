# Literature notes (Phase 1)

Date of search: **2026-10-24**. Companion to `01_research_plan.md` §0 (gap statement).

## Verification levels (read this first)

I could not open full PDFs for most papers. Notes below come from arXiv abstract pages or HTML fetched through a summarizing tool, and from search-result snippets. So:

- **abstract-verified**: arXiv abstract page fetched; metadata and abstract-level claims are reliable, details beyond the abstract are not.
- **html-verified**: arXiv HTML body fetched and summarized; method details reliable but not checked line by line by the author.
- **snippet-only**: only a search snippet seen. Do not cite for anything beyond "exists, topic is X".
- **not found**: could not locate. Do not cite.

**Before submission (P11.4) every cited paper must be opened and checked by the author against the source.** Nothing here replaces that.

## Papers

### Fork ledger — "Measuring the Value of World-Model Updates: A Counterfactual Utility Protocol for Continual Adaptation" (abstract-verified)
arXiv:2609.10954, Anqi Peter Li, Kaden Kim, submitted 10 Sep 2026, cs.LG, under review at the CWM workshop.
- Setting: continual world models on control tasks (CartPole, Walker, Cheetah).
- Measures: at pre-registered decision points the stream forks into an update continuation and a hold continuation (common random numbers); ΔR = R_update − R_hold on the same episodes. 240 attempted forks per task. A fixed update mechanism lowers return on all three tasks (−144.0, −82.8, −18.6).
- Argues triggers should be judged by the updates they select, not by surprise detection alone.
- Does NOT (per abstract): gate/commit/reject updates, check retention, or use LoRA/LMs. It is a **measurement protocol**, which is the source of our oracle ΔR idea.
- Still to read in full: how forks are matched, whether a rejection policy is discussed in the body.

### STABLE — "Gated Continual Learning for Large Language Models" (html-verified) — CLOSEST COMPETITOR
arXiv:2510.16089, William Hoy, Nurcin Celik, v1 17 Oct 2025.
- Setting: sequential self-edits of Qwen2.5-7B with LoRA merges on SQuAD-style factual edits; 8 sequential edits per run, 12 runs per gate.
- Gate: every candidate LoRA merge is checked against a stability budget (EM drop, bits increase, or KL divergence); on violation the update is clipped/rescaled or rejected.
- Check data: **anchor questions = earlier edited datapoints** (not a probe of unrelated old domains, not a held-out new-domain set). Thresholds are set manually (EM 7%/10%, bits 0.06/0.08, KL 0.5/0.7).
- No trigger (every candidate goes through the gate). No AUROC or comparison against a ground-truth oracle. No domain-shift text stream or perplexity. Baselines are other gating metrics and the unadapted model; EWC, replay, SEAL-ungated are discussed but not run.
- Stated limitations: gate overhead per merge; strict gates reject slightly-over-budget but useful edits; only factual edits tested.
- **Implication:** the idea "gate LoRA updates on a stability budget" is not new. Our contribution must rest on what STABLE lacks (see gap statement below). Must be cited and compared in related work.

### SEA — "Self-Evolving Agents with Anytime-Valid Certificates" (abstract-verified)
arXiv:2607.00871, Biswa Sengupta, 1 Jul 2026.
- Frozen base model; self-modification limited to a small steering adapter and versioned harness; each modification admitted only through an anytime-valid gate with an error budget. Evaluated on a 52-instance SWE-bench Verified subset.
- Does not (per abstract): describe a retention check, say whether the adapter is LoRA, or report variance. Different setting (agents/code), not continual text streams.
- Relevance: another "gate on modifications" paper; cite as adjacent. Needs full read before characterizing further.

### Engram Adapter — "When to Adapt: Conditional Memory Adapters for Retention-Preserving Domain Specialization" (abstract-verified)
arXiv:2608.29327, Jiayu Hou, Lei Wang, 29 Aug 2026, Findings of EMNLP 2026.
- Input-time selectivity (n-gram matching + learned scalar gate) on a frozen Qwen3 backbone; reports 99.4–100.1% retention of out-of-domain scores.
- Not an update-validation or rollback method; the gate acts per input at inference. Useful as retention-measurement precedent only.

### Continual-LoRA baselines
- **O-LoRA** — "Orthogonal Subspace Learning for Language Model Continual Learning", arXiv:2310.14152, Findings of EMNLP 2023 (snippet-only). Each new task in a low-rank subspace orthogonal to earlier tasks, no replay. Targets LMs, so most natural fit for text-stream B4 if it can be adapted to perplexity streams without task IDs/instructions (to check).
- **SD-LoRA** — arXiv:2501.13198, ICLR 2025 (snippet-only). Decouples LoRA magnitude and direction, rehearsal-free. Evaluated on class-incremental (vision) settings; code at github.com/WuYichen-97/SD-Lora-CL. Likely hard to port to LM streams.
- **C-LoRA** — "Continual Low-Rank Adaptation for Pre-trained Models", arXiv:2502.17920, Zhang, Bai, Yang (snippet-only). One shared LoRA with a learnable routing matrix. Name collides with unrelated papers (2304.06027 diffusion; 2505.17773 uncertainty); cite by arXiv ID.
- **Decision for B4 (P1.2), provisional:** use **plain replay-LoRA** as B4 (simple, fully reproducible, matches "replay" comparison in the plan); treat O-LoRA as optional second baseline if time allows. This needs the author's confirmation; none of the three has been read in full.

## Not found / not verified — do not cite

| Name in plan | Result |
|---|---|
| **CAWM** | No arXiv paper with this acronym found. Need title or authors from the author. |
| **SEGA** | No paper with this name found. Possibly meant STABLE or SEA above, or a different title. Need title from the author. |
| **SPREAD** (P1.4) | No paper found. **Drop from related work** unless the author supplies a reference. |
| **replay-retention paper** | Which paper this is was not specified in the plan; not identified. Need title. |

## Coverage matrix

Legend: Y = present per the source, N = absent per the source, ? = not stated in what I could read.

| Paper | Trigger | Predicts / measures update value | Commit / reject | Retention check | LoRA |
|---|---|---|---|---|---|
| Fork ledger 2609.10954 | evaluates triggers | Y (counterfactual ΔR) | N | N (same-task return) | N (world models) |
| STABLE 2510.16089 | N | N (no oracle comparison) | Y (clip/reject) | Y, but only on earlier-edit anchors | Y |
| SEA 2607.00871 | ? | ? | Y (certificate gate) | ? | ? (steering adapter) |
| Engram 2608.29327 | N (per-input gate) | N | N | Y (OOD benchmarks, post hoc) | N (not stated) |
| O-LoRA / SD-LoRA / C-LoRA | N | N | N | implicit (orthogonality, no probe) | Y |
| **CommitGate (this work)** | any (entropy, spike, fixed) | **Y (AUROC vs oracle ΔR)** | Y | Y (rotating probe + untouched final suite) | Y |

## P1.3 queries run (2026-10-24)

1. `CAWM continual adaptation world model arXiv` — not found
2. `SEGA continual learning LLM arXiv self-evolving gated adapter` — surfaced STABLE, SEA
3. `SPREAD continual learning LoRA arXiv` — not found
4. `O-LoRA orthogonal subspace learning continual learning language models arXiv`
5. `SD-LoRA scalable decoupled LoRA class incremental learning arXiv`
6. `C-LoRA continual low-rank adaptation arXiv`
7. `"commit gate" OR "update validation" rollback candidate LoRA adapter continual learning language model held-out retention probe` (extended) — no exact match; surfaced Engram, Baseten rank-1 LoRA rollback post (blog, not a paper)
8. `papers citing "Measuring the Value of World-Model Updates" counterfactual utility fork ledger` — no citing papers found (search engine is not a citation index; use Google Scholar / Semantic Scholar "cited by" manually)
9. `entropy-triggered LoRA adapter growth low-resource languages continual adaptation` — no match for the author's prior method

Limits: a general web search is not a citation index. The fork-ledger citation check is **incomplete** and must be repeated by hand on Google Scholar / Semantic Scholar, and again before submission (P11.3).

## Gap statement check (against `01_research_plan.md` §0)

The claim "almost none decide whether to keep a candidate update" is **too strong**: STABLE gates LoRA merges with a stability budget. It still holds in narrower form. What STABLE does not do, per what I read:
1. no gain-on-new-data term (its check is stability only, so it cannot trade gain against cost: no Score = gain − λ·cost);
2. retention checked only on earlier edits, not on a held-out retention probe from past domains, and no untouched final suite;
3. no predictive-validity analysis (gate AUROC / correlation vs oracle ΔR);
4. no study of probe size, probe rotation/overfitting, or threshold calibration on a separate validation stream;
5. no trigger interplay (gate on top of entropy / spike / fixed triggers);
6. factual-edit setting only; no domain-shift text streams (abrupt, gradual, recurring, noise).

Proposed rewording (needs author approval, then log in `decisions.md`): "Existing gated-update work checks stability of LoRA merges against a budget, but does not score candidates on both held-out new-domain gain and retention, and does not measure how well such a gate predicts the true value of an update." Sharpen the paper around items 1, 3, 4 and 5; **item 3 (predictive validity vs oracle ΔR) is the strongest differentiator** and borrows the measurement idea from the fork ledger.

## Open items for the author

- Titles/authors for CAWM, SEGA, and the replay-retention paper.
- Confirm B4 choice.
- Open the full text of STABLE, fork ledger and SEA (the three that matter most) and confirm the characterizations above.
- Manual citation search for the fork ledger on Google Scholar / Semantic Scholar.
