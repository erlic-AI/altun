---
license: apache-2.0
language:
- en
- tr
library_name: pytorch
pipeline_tag: text-generation
tags:
- math
- theoretical-computer-science
- from-scratch
- research-prototype
---

# ALTUN 1 (v0.1)

A 197M-parameter decoder-only language model for mathematics and theoretical computer science, **trained from scratch** (no pretrained weights) by erlic AI on Kaggle 2×T4 GPUs. Code, training pipeline and evaluation: https://github.com/erlic-AI/ALTUN-1

**This is a small research prototype, not a reliable math solver.** It often produces fluent step-by-step solutions that contain wrong calculations.

## Model

- Decoder-only Transformer: 16 layers, d_model 1024, 16 heads / 4 KV heads (GQA), SwiGLU (intermediate 2816), RoPE, RMSNorm, tied embeddings
- 197,166,080 parameters, context length 1,024, vocabulary 16,384 (own byte-level BPE with math-aware pre-tokenization)
- Files: `model.safetensors`, `config.json`, `tokenizer/`
- **Custom architecture: it does not load with `transformers`.** Use the code in the GitHub repository.

## How to use

```bash
git clone https://github.com/erlic-AI/ALTUN-1 && cd ALTUN-1
pip install torch safetensors pyyaml psutil huggingface_hub
huggingface-cli download erlic/ALTUN-1 --local-dir ALTUN-1-model
python -m inference.run_inference --model-dir ALTUN-1-model --prompt "What is 12 + 15?"
```

Prompt format: `Problem: <question>\nSolution:` (added automatically by the CLI). Use greedy decoding; sampling with a repetition penalty makes arithmetic worse.

## Training data

About 1.94B training tokens over three sequential runs (each continued from the previous weights): Mathlib4, Lean 4 core, Metamath set.mm, TheAlgorithms/Python, HoTT book, English and Turkish Wikipedia (math/CS categories), NuminaMath-CoT, MetaMathQA, GSM8K train, MATH train, and programmatically generated, verifiable step-by-step problems (EN + TR, ~30% Turkish). Each source passed a license gate (allow-list), exact and near-duplicate removal, and 13-gram decontamination against GSM8K/MATH test data and the ALTUN evaluation set. Per-source licenses and caveats: see [NOTICE](https://github.com/erlic-AI/ALTUN-1/blob/main/NOTICE).

## Training

2×T4 (DDP, FP16 AMP), AdamW, wall-clock-based cosine LR (run 3: peak 2.5e-4), 131,072 tokens/step, `torch.compile` auto-enabled after a measured 1.39× speedup. Run 3: 4,291 steps, 562M tokens, ~22.4k tokens/s, no skipped steps, final val loss 0.645. Loss values are not comparable across runs because the data mix changed.

## Evaluation (measured, greedy, 256 new tokens)

| Benchmark | Result | Caveat |
|---|---|---|
| ALTUN custom set, 119 questions / 13 categories | 53.8% (64/119) | **In-distribution**: training generators target the same skills; small set, high variance |
| GSM8K, first 200 test questions | **1.5%** (3/200) | independent |
| MATH, 200 test questions (sampled) | **5.0%** (10/200) | independent |
| Turkish mini set, 32 hand-written questions | 25.0% (8/32) | tiny set |
| AIME, MiniF2F | not run | no data / no Lean verifier |

ALTUN categories: Arithmetic 81%, Formal Mathematics 83%, Proof Completion 80%, Number Theory 67%, Algorithm Analysis 62%, Linear Algebra 56%, Combinatorics 56%, Computational Complexity 57%, Algebra 44%, Graph Theory 44%, Calculus 33%, Mathematical Logic 22%, Algorithms 0%. Full table: `results/evaluation_summary.md`.

## Limitations and intended use

- Intended for research, education and as a base for further from-scratch experiments, not for any task needing correct answers.
- Weak on word problems, competition math and Turkish; 1,024-token context; English and Turkish only.
- Can produce plausible but false proofs and calculations. It makes no claim about unsolved problems (e.g. P vs NP).
- Trained on small data compared with modern LLMs; no safety fine-tuning.

## License

Apache-2.0. Training data includes CC-BY-SA material (Wikipedia, HoTT book) and machine-generated QA; whether weights trained on share-alike text are derivative works is legally unsettled. See NOTICE; this is not legal advice.

## Citation

```
@misc{altun1,
  title  = {ALTUN 1: a small from-scratch math and theoretical-CS language model},
  author = {erlic AI},
  year   = {2026},
  url    = {https://github.com/erlic-AI/ALTUN-1}
}
```
