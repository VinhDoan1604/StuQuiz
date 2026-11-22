# OCR vs VLM on synthetic scans (`ocr-v1`)

10 synthetic scans of Vietnamese lecture notes (levels: light 4, medium 3, heavy 3). ↓ lower is better, ↑ higher is better. Brackets: 95 % bootstrap CI.
Tesseract lang `vie+eng`; VLM engines are labelled by model (task `vision`, same prompt). Commit `80e2100-dirty`.

| Engine | CER ↓ | CER without diacritics ↓ | WER ↓ | Word F1 ↑ (order-free) | Errors | s / page (median) | Tokens / page (mean) |
|---|---|---|---|---|---|---|---|
| `tesseract` | 0.066 [0.058–0.073] | 0.049 [0.040–0.057] | 0.202 [0.188–0.219] | 0.826 [0.810–0.839] | 0/10 | 3.2 | 0 |
| `vlm:gemini-2.5-flash` | 0.046 [0.013–0.090] | 0.044 [0.012–0.086] | 0.085 [0.042–0.139] | 0.964 [0.955–0.973] | 0/10 | 19.7 | 1,873 |

**By degradation level** (mean CER; few images per level, so no CI):

| Engine | light | medium | heavy |
|---|---|---|---|
| `tesseract` | 0.063 | 0.066 | 0.068 |
| `vlm:gemini-2.5-flash` | 0.012 | 0.107 | 0.031 |

**Paired difference, `vlm:gemini-2.5-flash` − `tesseract`** (error rates: negative = better; word F1: positive = better):

- `cer`: -0.019 [-0.052, +0.024], n = 10 → no demonstrated difference
- `cer_nodiac`: -0.005 [-0.036, +0.035], n = 10 → no demonstrated difference
- `wer`: -0.117 [-0.162, -0.059], n = 10 → vlm:gemini-2.5-flash better
- `word_f1`: +0.138 [+0.125, +0.154], n = 10 → vlm:gemini-2.5-flash better

Caveats: synthetic degradation of digital pages, not real scans or phone photos; one page set from one course; ground truth is the PDF text layer. Per-image rows are in `raw.jsonl` (not published: the source notes may not be redistributed).
