# Design decisions

This file exists because `EXPERIMENT_GUIDE_encoder_ambiguity.md` §7 requires
every gap the paper leaves unfilled to be decided **once**, written down, and
reused identically across both encoder strategies and all three sizes — so a
result gap can always be attributed to the one thing the experiment is
actually testing (`conv_on_patches` vs `conv_on_image`), never to an
accidental difference in something that should have been held constant.

Paper: Agarwalla, Kumar, Dandekar, Dandekar, Panat, *NanoVLMs: How small can
we go and still make coherent Vision Language Models?*, arXiv:2502.07838.

---

## 1. Caption format: single format, not ShortDesc + LongDesc

The paper trains six models total: {mini, base, large} × {ShortDesc (20–25
words), LongDesc (60–70 words)}. Neither guide handed to this build mentions
generating two caption datasets — both describe exactly one 28K-pair
generation pass — and `DGX_GUIDE_nanovlm.md` §5 names "six checkpoint/result
pairs total," which only adds up under one description length, not two
(that would be twelve).

**Decision:** generate a single caption style, modeled on the paper's
LongDesc (60–70 words) rather than ShortDesc. Rationale: a longer, richer
description exercises the visual encoder harder — more of the image has to
be reflected in the generated text — which makes it a more discriminating
test of "does the encoder reading matter" than a 20-word caption where most
architectural differences would be washed out by how little text there is
to get right. This is a deliberate scope decision, not an oversight; if the
encoder question turns out to matter, re-running with ShortDesc as a second
axis is a natural follow-up experiment, not a rebuild.

Consequence: the held-out evaluation's partial-text prompt (§6 below) also
uses the paper's *long* partial-text length (18–20 words), consistent with
long-format training captions.

---

## 2. Conv hyperparameters — `conv_on_patches` (Section 2.2.1's text)

Per-patch pipeline, each 16×16×3 patch processed independently (patches
extracted via `unfold`, batched as `(B*196, 3, 16, 16)`):

| Layer | Kernel | Stride | Padding | Channels |
|---|---|---|---|---|
| Conv2D + ReLU | 3×3 | 1 | 1 | 3 → 32 |
| Conv2D (stride=2) | 3×3 | 2 | 1 | 32 → 64 |

16×16 → (stride-1 conv, same size) → 16×16 → (stride-2 conv) → 8×8.
Flatten to `64 * 8 * 8 = 4096` → LayerNorm(4096) → ReLU → Linear(4096,
`img_embd_dim`). Output: `(B, 196, img_embd_dim)`.

These channel counts (32, 64) are fixed **globally** — not scaled per model
size — matching the paper's pattern where only `n_blks`/`n_layer`/`n_head`/
`head_size`/`n_embd`/`img_embd_dim` vary by size (Table 1) while everything
else (dropout, image size, patch size, lr) is fixed across sizes (§2.3).

## 3. Conv hyperparameters — `conv_on_image` (Figure 4's reading)

Whole-image CNN stem, depth computed from config rather than hardcoded:

| Layer | Kernel | Stride | Padding | Channels | Spatial |
|---|---|---|---|---|---|
| Stem: Conv2D + ReLU | 3×3 | 1 | 1 | 3 → 32 | 224×224 |
| Conv2D + ReLU | 3×3 | 2 | 1 | 32 → 64 | 112×112 |
| Conv2D + ReLU | 3×3 | 2 | 1 | 64 → 128 | 56×56 |
| Conv2D + ReLU | 3×3 | 2 | 1 | 128 → 256 | 28×28 |
| Conv2D + ReLU | 3×3 | 2 | 1 | 256 → 256 | 14×14 |
| `AdaptiveAvgPool2d(14)` | — | — | — | 256 | 14×14 (no-op here) |

Four stride-2 layers after the stem: 224→112→56→28→14, exactly matching
`DGX_GUIDE_nanovlm.md`'s stated arithmetic for the paper's 224/16 defaults.
The adaptive pool is unconditional (a safety net for any future config where
the stride-2 arithmetic doesn't land exactly on 14×14) and a no-op for the
current image/patch size.

Output `(B, 256, 14, 14)` is reshaped to `(B, 196, 256)` — one token per
spatial location, channel vector as its feature — then LayerNorm(256) →
ReLU → Linear(256, `img_embd_dim`), giving `(B, 196, img_embd_dim)`,
**matching `conv_on_patches`'s output shape exactly** as required.

Channel counts are shared with `conv_on_patches` where the layer roles
overlap (32→64 first two conv layers) so that the two strategies diverge
only in *whether* convolution runs per-patch or over the whole image, not in
arbitrarily different channel-width choices.

**Parameter count is allowed to differ between the two strategies** (per
`EXPERIMENT_GUIDE_encoder_ambiguity.md` §2) — `conv_on_image` has a deeper,
wider conv stack than `conv_on_patches`. Both architectures' total parameter
counts are recorded per size in every `results/*_eval.json` and must be
reported alongside judge scores, never silently.

---

## 4. Shared transformer block (both encoder and decoder, Figure 5)

Pre-LN transformer block: `x = x + Attn(LN(x))`, `x = x + MLP(LN(x))`.
MLP inner dimension = **4× the block's embedding dim** (standard transformer
default; the paper doesn't specify an expansion ratio). Dropout = 0.1
everywhere a dropout is applicable (attention weights, MLP, embeddings) —
the one dropout value the paper does fix (§2.3), applied uniformly since it
gives no per-location breakdown.

### Attention head sizing

Table 1 gives `n_head` and `head_size` such that `n_head * head_size ==
n_embd` exactly for all three sizes (mini: 8×12=96, base: 8×16=128, large:
16×12=192) — so those two columns describe the **decoder's** attention
(operating on the `n_embd`-dim text stream), and the decoder uses `head_size`
literally as given in Table 1.

The paper gives no separate head-sizing for the **visual encoder**
(operating on `img_embd_dim`, which is not `n_embd`). Decision: the encoder
reuses the same `n_head` as the decoder for that size, with per-head
dimension derived as `img_embd_dim // n_head` (the standard way to
parameterize attention when only head *count* and total embedding width are
given). This divides evenly for all three sizes: mini 400/8=50, base
512/8=64, large 512/16=32 — no rounding needed, which is itself a small
confirmation this is the intended reading.

### CLS token / visual representation

Encoder: patch embed → prepend learnable `[CLS]` (197 tokens total) →
learnable positional embedding → LayerNorm → `n_blks` transformer blocks →
take the `[CLS]` token's output as the image's compact representation
(`img_embd_dim`-dim vector), per §2.2.1's own description ("the `[CLS]`
token is aggregated to form a compact representation").

### Visual-textual connector

Single learnable `Linear(img_embd_dim, n_embd)` + GELU (§2.2.2: "a single
learnable layer followed by GELU"), applied to the CLS vector, producing one
visual token in the text embedding space. That token is **prepended** to
the text token embedding sequence before the decoder's positional embedding
is applied — the natural reading of "both the visual and textual embeddings
are concatenated to form a multimodal token embedding" for a single-vector
visual representation feeding an autoregressive decoder.

### Decoder

Standard causal (masked) transformer, `n_layer` blocks as sized above, final
LayerNorm → `Linear(n_embd, vocab_size)`. Cross-entropy loss over text
positions only (the prepended visual-token position is excluded from the
loss — it has no target token).

---

## 5. Tokenizer

Byte-level BPE (via Hugging Face `tokenizers`), **vocab size 8000** — small
deliberately: the training corpus is child-simple synthetic captions with a
narrow, repetitive vocabulary (the entire premise of the paper), and these
models are 5M–25M parameters total, so a large subword vocabulary would
waste a disproportionate share of parameters on the embedding/output tables.
Special tokens: `<pad>`, `<bos>`, `<eos>`, `<unk>`.

Trained **once**, on the **training split's captions only** (not
validation, not held-out) — standard practice to keep held-out evaluation
uncontaminated — and reused unmodified by both architectures, per
`EXPERIMENT_GUIDE_encoder_ambiguity.md` §2's constancy table.

---

## 6. Held-out evaluation set: 100 images, not the paper's 25

Per `EXPERIMENT_GUIDE_encoder_ambiguity.md` §1: the paper's own 25-sample
judge evaluation is a known-risky size — an earlier project on this same
line of work found 25 held-out samples produced a trend that reversed at
100. **100 held-out images**, selected once (seed 42, before the
train/val split, so it's the same 100 images regardless of how the
remaining ~27,900 get split 90/10), reused identically across all 6
checkpoint evaluations. This is a deliberate deviation from the paper for
statistical reliability — noted explicitly here and in every results
write-up, not a silent substitution.

Partial-text completion prompts for evaluation use the paper's long-format
length (18–20 words), consistent with the LongDesc-style training data
(§1 above).

---

## 7. Random seed policy

**Seed 42**, fixed, reused for: COCO image ID selection (held-out first,
then train/val split on the remainder — `scripts/download_coco.py`),
tokenizer training, and model weight initialization for **both**
architectures at every size. The only thing that varies between a
`conv_on_patches` run and a `conv_on_image` run at the same size is the
`--patch-embed-strategy` flag — same data, same tokenizer, same init seed —
so a score gap is attributable to the encoder question being asked, not to
noise from different random draws.

This is a **single-seed study** (one run per size × strategy combination),
not a multi-seed average — a real caveat on any conclusion drawn from it,
called out again in `EXPERIMENT_GUIDE_encoder_ambiguity.md` §6 and repeated
here rather than left as an unstated assumption.

---

## 8. Epoch count / batch size

**20 epochs, batch size 64, for all three sizes** — carried over from the
earlier project's `base`-size numbers rather than re-tuned per size. This
is a scope decision, not a validated-optimal one: `mini` may plateau well
before 20 epochs and `large` may benefit from more, but per-size tuning
would introduce yet another axis of asymmetry between the two architecture
arms (if `mini_conv_on_patches` trains for a different epoch count than
`mini_conv_on_image`, an eval gap is confounded again). Same epoch/batch
budget for both strategies at a given size keeps the comparison clean;
revisiting per-size epoch counts is a reasonable follow-up once the first
full run's loss curves are in hand (watch for `mini` clearly plateauing
early or `large` still improving at epoch 20 in the logs).

Learning rate: `1e-3` (paper §2.3, fixed across all sizes).

---

## 9. `--mem` / `--time` sbatch values

**Not filled in with real numbers** — `EXPERIMENT_GUIDE_encoder_ambiguity.md`
§7 / `DGX_GUIDE_nanovlm.md` §6 both require these to come from a real timing
measurement on *this* cluster, *this* repo's actual code, not inherited or
guessed. The `dgx/*.sbatch` scripts in this repo carry placeholder values
marked `# TODO(bench):` with the `shortq` diagnostic command to run first —
filling these in requires actually running on `dgx1`, which this build
session cannot do. Do not raise these past the placeholder without running
that diagnostic; do not trust the placeholder for a real submission either.

---

## 10. Storage / naming

No deviation from `DGX_GUIDE_nanovlm.md` §2 and §5 — reused as specified:
`data/processed/nanovlm_28k/{train,val,eval_holdout}.jsonl`,
`checkpoints/nanovlm_<size>_<strategy>/final.pt`,
`results/nanovlm_<size>_<strategy>_eval.json`,
`logs/nanovlm-<stage>-<jobid>.out/.err`, generic Slurm job names (not
containing "nanovlm").
