# GPT-2 Domain Adaptation on 19th-Century Fiction (Full FT vs LoRA)

This project builds an end-to-end*domain adaptation pipeline for causal language modeling using a cleaned Project Gutenberg novel (The Leavenworth Case, Anna Katharine Green). It benchmarks:

- Pretrained GPT-2 (baseline perplexity + generation)
- Full fine-tuning (validation-driven training + best checkpoint)
- LoRA fine-tuning (PEFT) for parameter-efficient adaptation

It also includes lightweight generation quality metrics, a simple memorization proxy, and a “stable decoding” configuration for more controlled generation.

---

## Highlights (Engineering-Focused)

- Reproducible train/val/test split at paragraph level, then blockwise chunking for causal LM.
- Baseline generation + baseline test perplexity evaluation (same held-out test blocks).  
- Full fine-tuning with step-based eval + checkpointing + early stopping.  
- LoRA training with PEFT to compare quality vs efficiency.
- Automated outputs saved under a single run directory.

---

## Data Source

- Novel: [*The Leavenworth Case*](https://www.gutenberg.org/ebooks/4047) by Anna Katharine Green
- Downloaded as UTF-8 plain text, then stripped of Gutenberg boilerplate and front-matter (contents/illustrations), and trimmed to story start.

---

## Pipeline Overview

### 1) Preprocessing
- Strip Gutenberg header/footer via delimiter regex.
- Remove illustrations/contents sections.
- Trim to a story anchor ("BOOK I. THE PROBLEM").
- Normalize whitespace and paragraph breaks.

### 2) Dataset Split
- Split into paragraphs (drop very short paragraphs to stabilize training).
- Random split with fixed seed into:
  - train / validation / test = 80% / 10% / 10%

### 3) Tokenization & Chunking
- `GPT2TokenizerFast` with an explicit `[PAD]` token to avoid padding being treated as EOS.
- Concatenate tokens then chunk into fixed-length blocks (default `block_size=128`).

### 4) Baseline Evaluation (Pretrained GPT-2)
- Evaluate pretrained GPT-2 on the same `lm["test"]` blocks and report perplexity.
- Generate continuations on a shared prompt set.

### 5) Full Fine-tuning
- Train GPT-2 with step-based evaluation, checkpointing, and early stopping.
- Evaluate on held-out test blocks and report perplexity.

Full FT configuration uses:
- `eval_strategy="steps"`, `eval_steps=25`
- `save_strategy="steps"`, `save_steps=25`
- `load_best_model_at_end=True`
- early stopping patience = 3  
(see implementation)  

### 6) LoRA (PEFT)
- Re-load base GPT-2, apply LoRA adapters on attention/projection modules.
- Train and evaluate the LoRA model on the same split.

### 7) Generation & Reporting
- Build a 20-prompt set (10 extracted from test paragraphs + 10 handcrafted prompts).
- Generate baseline vs fine-tuned outputs with identical decoding settings.
- Add a “stable decoding” variant using `no_repeat_ngram_size=3`.

### 8) Metrics
- Distinct-2 / Distinct-3
- 4-gram repetition rate
- Average length (tokens)
- Memorization proxy: fraction of generated 8-grams that appear in the train set

---

## Key Result
### Held-out Test Perplexity (same `lm["test"]` blocks)
| Method | Test Loss | Test PPL |
|---|---:|---:|
| Pretrained GPT-2 (baseline) | 4.1449 | 63.11 |
| Full fine-tuning | *3.3242* | *27.78* |
| LoRA (r=8) | 3.5220 | 33.85 |

**Impact**
- Full fine-tuning reduced test perplexity from **63.11 → 27.78**.
- LoRA reduced test perplexity from **63.11 → 33.85**, while training only **0.65%** of parameters (0.81M / 125M).

### Generation Quality (20-prompt set)
| Metric | Baseline | Full FT | Change |
|---|---:|---:|---:|
| Distinct-2 | 0.9415 | 0.9315 | −0.0100 |
| Distinct-3 | 0.9954 | 0.9959 | +0.0005 |
| 4-gram repetition rate | 0.00084 | 0.00000 | ↓ to zero |
| Avg length (tokens) | 121.6 ± 16.5 | 124.4 ± 9.0 | longer + more stable |

**Interpretation**
- Full fine-tuning maintained high lexical diversity (Distinct-3 ~0.996) while **eliminating 4-gram repetition** under this evaluation.
- Generated length became **more consistent** (std **16.5 → 9.0**), indicating more stable decoding behavior on the prompt set.

