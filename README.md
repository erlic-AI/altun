<div align="center">

# ALTUN 1

### Compact Adaptive Computation Research

**erlic AI · Independent AI Research**

[Hugging Face](https://huggingface.co/erlic/ALTUN-1) ·
[Results](https://github.com/erlic-AI/altun/results/evaluation_summary.md)

**ALTUN 1.0.0 — First public release**

</div>

---

**A small (197M-parameter) decoder-only language model for mathematics and theoretical computer science, trained from scratch on 2×T4 GPUs by [erlic AI](https://github.com/erlic-AI).**

> **Status: v0.1 research prototype.** It writes step-by-step solutions in a fixed format and is decent at simple arithmetic and short formal/proof-completion questions. It is **not** a reliable calculator or solver: on independent benchmarks it scores **1.5% on GSM8K and 5.0% on MATH** (details below). It makes no claim about open problems such as P vs NP.

Model weights: **https://huggingface.co/erlic/ALTUN-1** · Turkish guide: [README.tr.md](README.tr.md) · Model card: [MODEL_CARD.md](MODEL_CARD.md)

## What is in this repository

A complete, from-scratch engineering stack (no pretrained weights anywhere):

| Part | Where |
|---|---|
| Transformer (RMSNorm, RoPE, GQA, SwiGLU, tied embeddings, KV-cache generation) | `altun/` |
| Own byte-level BPE tokenizer (math-aware pre-tokenization, 16,384 vocab) | `altun/tokenizer.py` |
| Data pipeline: license gate, cleaning, exact + MinHash near-dedup, benchmark decontamination, train/val/test split | `data/` |
| Verifiable synthetic math generator (arithmetic, algebra, number theory, graphs, calculus, Lean, ...; EN + TR) | `data/synthetic_math.py` |
| Training: 2-GPU DDP, FP16, wall-clock cosine LR, atomic checkpoints + resume, `torch.compile` with measured fallback | `training/` |
| Evaluation: 119-question ALTUN set, Turkish mini set, GSM8K/MATH test samples | `evaluation/` |
| One-command launcher (continues from the Hugging Face weights; any machine with an NVIDIA GPU) | `start.py`, `scripts/run_training.py` |

## Training history (all numbers measured on Kaggle 2×T4)

| Run | New tokens | Notes |
|---|---|---|
| 1 | 625M | from scratch |
| 2 | 757M | continued from run 1, math-QA + synthetic-heavy mix |
| 3 | 562M | continued from run 2, more synthetic data for weak categories, higher Turkish share |
| **Total** | **≈1.94B** | final val loss 0.645 (mixes differ between runs, so losses are not comparable across runs) |

## Results (v0.1, greedy decoding, 256 new tokens, prompt `Problem: ...\nSolution:`)

| Benchmark | Result | Caveat |
|---|---|---|
| ALTUN custom set (119 questions, 13 categories) | **53.8%** (64/119) | The training generators target the same skill areas, so this set is *in-distribution* and flatters the model. Small set, high variance. |
| GSM8K (200 test questions) | **1.5%** (3/200) | independent, test split excluded from training |
| MATH (200 test questions) | **5.0%** (10/200) | independent, test split excluded from training |
| Turkish mini set (32 hand-written questions) | **25.0%** (8/32) | tiny set; Turkish is weak |
| AIME, MiniF2F | **not run** | no data file / no Lean verifier available |

Per-category numbers (e.g. Arithmetic 81%, Formal 83%, but Algorithms 0% and Logic 22%) are in [results/evaluation_summary.md](results/evaluation_summary.md). Wrong answers are listed unfiltered in the training outputs.

## Quick start (inference)

```bash
git clone https://github.com/erlic-AI/ALTUN-1 && cd ALTUN-1
pip install torch safetensors pyyaml psutil
pip install -U huggingface_hub
huggingface-cli download erlic/ALTUN-1 --local-dir ALTUN-1-model
python -m inference.run_inference --model-dir ALTUN-1-model --prompt "What is 12 + 15?"
```

The model expects the format `Problem: <question>\nSolution:` (the CLI adds it for you; use `--raw` to disable). Use greedy decoding (the default) for math. It is a custom architecture, so it does **not** load with `transformers`; use the code in this repo.

## Reproduce / continue training (any machine with NVIDIA GPU(s); tested on 2×T4)

```bash
git clone https://github.com/erlic-AI/ALTUN-1 && cd ALTUN-1
pip install -r requirements.txt        # already present on Kaggle
python start.py                        # continues from the released weights, 7 hours
```

`start.py` downloads the weights from Hugging Face, fetches the open-licensed training data, runs hardware preflight checks, trains, evaluates, and writes everything to `./ALTUN-1-RESULTS/` (on Kaggle: `/kaggle/working/ALTUN-1-RESULTS/`).
Useful options: `--hours 10`, `--continue-from none` (train from scratch), `--continue-from /path/to/final_model`, `--smoke` (tiny test run), `--output-dir`.
It automatically uses all visible GPUs (2×T4 DDP in our runs). On a **free Kaggle notebook** (GPU T4 ×2, Internet on) run in a cell: `!git clone https://github.com/erlic-AI/ALTUN-1 && cd ALTUN-1 && python start.py`, then *Save & Run All* to keep the outputs. The training code was run end-to-end only on 2×T4; other setups are untested.

Full Turkish instructions: [README.tr.md](README.tr.md). Technical report: [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md). Test report (what was and was not tested): [docs/TEST_REPORT.md](docs/TEST_REPORT.md).

## Limitations (please read)

- Frequently produces confident but wrong calculations; do not use for anything that needs correct answers.
- Weak on word problems (GSM8K 1.5%), competition math (MATH 5%), and Turkish (25% on a tiny set).
- Context length 1,024 tokens. English and Turkish only.
- The 119-question ALTUN set is in-distribution for the training generators; independent numbers are the GSM8K/MATH ones.
- Training data includes CC-BY-SA text (Wikipedia, HoTT book) and machine-generated QA; see [NOTICE](NOTICE).

## License

Code and weights: Apache-2.0 (see [LICENSE](LICENSE)), with the data attribution and caveats in [NOTICE](NOTICE). This is not legal advice.
