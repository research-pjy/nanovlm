# Experiment: Resolving the paper's visual encoder ambiguity

Companion to `DGX_GUIDE_nanovlm.md` (infra/how-to-run). This doc is the
research design: what the ambiguity actually is, what the two
architectures are, what stays identical between them, and how to read
the results. Give both docs to the Cowork build session.

---

## 0. What this experiment is (and isn't)

The NanoVLMs paper (Agarwalla, Kumar, Dandekar, Dandekar, Panat,
arXiv:2502.07838) describes its visual encoder's patch-embedding step
two different ways, and includes no code:

- **Section 2.2.1's text**: image → split into 16×16 patches → each
  patch through two Conv2D layers → LayerNorm → ReLU → FC → 196 patch
  tokens. Reads as convolution applied *per patch*.
- **Figure 4**: looks like a *whole-image* CNN stem — one image goes in,
  a stack of conv layers runs over the full 224×224 image, producing
  feature maps that are then flattened to an output layer.

These aren't a wording difference — they're two different architectures.
**This experiment is the resolution**: build the text's literal reading
first, then the figure's reading, train both under identical conditions,
compare. There is no third "our architecture" here — both arms *are* the
paper, under its two possible readings. Whichever the results favor (or
if they're roughly tied) becomes the one you'd call "the replication"
going forward.

---

## 1. Dataset — 28K pairs, matching the paper's stated scale

Paper: "we selected approximately 28K image-caption pairs, using 90
percent for training and 10 percent for validation... tested on 25
separate data samples that were entirely distinct from the training and
validation sets" (Section 2.1.1).

**Decisions for this run:**

- **Total**: 28,000 image-caption pairs from COCO.
- **Source**: `val2017` (5000 images) is not enough on its own — pull the
  remainder from `train2017`. Both splits' `captions_*.json` give five
  captions per image; combine all five per image into the teacher prompt,
  same as the paper's own dataset prep (Figure 3).
- **Split**: 90/10 train/val on the 28K, matching the paper exactly
  (~25,200 train / ~2,800 val).
- **Held-out eval set size — open decision, not paper-matched by
  default.** The paper used 25 held-out samples for its GPT-4o judge
  evaluation. A prior experiment on the old project (prompt-vs-length,
  full-scale run) found that 25 was small enough to produce a spurious
  trend that reversed at 100 held-out samples — worth reading as a
  caution against reusing 25 here too. Recommend **100 held-out images**
  instead of the paper's 25, selected once and reused identically across
  all 6 checkpoint evaluations (see §4). This is a deliberate deviation
  from the paper for statistical reliability, not an oversight — note it
  explicitly wherever results get written up.
- **Selection**: fixed seed (42) for which images land in
  train/val/held-out, so the split is reproducible and — critically —
  **identical** across both architecture runs. Held out first
  (pre-`--limit` selection), then train/val split on the remainder, same
  pattern as the earlier project used.
- **Download script**: fetch `captions_train2017.json` +
  `captions_val2017.json` on the DGX **login node** (has internet), pick
  28K image IDs deterministically (seed 42), download only those specific
  files (`http://images.cocodataset.org/{split}/<filename>`) rather than
  the full ~18GB `train2017` zip — this keeps the download to roughly
  2-3GB instead of 18GB. Skip-existing (idempotent, resumable), write
  incrementally, same shape as the old `download_coco.py`.

---

## 2. What stays identical between the two architecture arms

This is the part that makes the comparison meaningful — if anything
besides the encoder differs, a score gap can't be attributed to the
architecture question the experiment is actually asking.

| Held constant | How |
|---|---|
| Training data | Same 28K-pair dataset, same train/val/held-out split (§1) |
| Tokenizer | Trained **once** on the generated caption corpus, reused by both architectures — neither's vocabulary differs from the other's |
| Teacher/grader model | `llama3:8b` for both generation and evaluation (see `DGX_GUIDE_nanovlm.md` §3 for why not `llama3:70b`) |
| Model size configs | Same mini/base/large hyperparameters (Table 1: `n_blks`, `n_layer`, `n_head`, `n_embd`, `img_embd_dim`) — only `patch_embed_strategy` changes |
| Epochs / batch size | Same per size, across both architectures |
| Random seed | Same seed for data split, tokenizer training, and model init |
| Evaluation prompt / rubric | Same grading prompt, same 5-dimension rubric (grammar, creativity, consistency, meaningfulness, plot) |

**What's allowed to differ** (and is genuinely informative, not noise):
total parameter count. `conv_on_image`'s downsampling depth is computed
from `image_size`/`patch_size`/conv hyperparameters rather than fixed, so
the two strategies are not guaranteed to land on the same encoder
parameter count for a given size config — **report this explicitly
alongside judge scores**, so a win (if any) isn't mistaken for "just more
parameters."

---

## 3. The two architectures

Both live behind one config flag (`patch_embed_strategy`), so nothing
else in the encoder (CLS token, positional embedding, transformer
blocks) changes between them — only how the initial `(B, 3, 224, 224)`
image becomes `(B, 196, img_embd_dim)` patch tokens differs.

### `conv_on_patches` (Section 2.2.1's text, build this first)

Each 16×16 patch, unfolded and processed independently:
Conv2D → ReLU → Conv2D(stride=2) → flatten → LayerNorm → ReLU → Linear →
`img_embd_dim`.

### `conv_on_image` (Figure 4's reading, build second)

One CNN stem over the whole 224×224 image (Conv2D → ReLU), followed by a
stack of stride-2 Conv2D+ReLU layers whose depth is *computed from
config* (not hardcoded) — for the paper's stated defaults (224×224 image,
16×16 patch) this should land on 224→112→56→28→14 in four layers. An
`AdaptiveAvgPool2d` to 14×14 runs unconditionally afterward as a safety
net (a no-op when the stride-2 arithmetic already lands exactly on
14×14; guarantees the right shape for any other config where it
doesn't). Flatten to 196 tokens → LayerNorm → ReLU → Linear →
`img_embd_dim`, matching the other strategy's output shape exactly.

**Conv hyperparameters** (kernel size, stride, padding, intermediate
channel count) are unspecified by the paper for *both* readings. Decide
these once, write them down in this repo's own `DESIGN_DECISIONS.md`,
and **reuse the identical values across both strategies** — if kernel/
stride differ between the two architectures too, a result gap can't be
cleanly attributed to "per-patch vs. whole-image," since the confound
(different conv hyperparameters) is now tangled with the actual question
being asked.

---

## 4. Running it (see `DGX_GUIDE_nanovlm.md` §4 for the sbatch commands)

Order, fully manual, no auto-chaining between stages:

1. **Generate** the 28K dataset once (self-chains internally to fit the
   time cap — see infra guide).
2. **Train `conv_on_patches`** at mini, base, large — three separate
   manual submissions.
3. **Evaluate `conv_on_patches`** at mini, base, large — three separate
   manual submissions, only after all three training runs are done.
4. **Train `conv_on_image`** at mini, base, large.
5. **Evaluate `conv_on_image`** at mini, base, large.

No interleaving between architectures, and train-all-three-then-eval-
all-three within an architecture (not train→eval→train→eval per size) —
review the full picture for one architecture before starting the other.

---

## 5. Comparing results

A generic comparison script (reads any `evaluate.py --output` JSON,
takes N files + labels) is worth writing once and reusing for all 6
pairwise/multi-way comparisons:

```bash
python experiments/encoder_ambiguity/compare_results.py \
    results/nanovlm_mini_conv_on_patches_eval.json \
    results/nanovlm_mini_conv_on_image_eval.json \
    --labels conv_on_patches conv_on_image
```

Run once per size (mini, base, large) — three comparisons total, plus
worth looking at whether the *direction* of any effect is consistent
across sizes or only shows up at one scale.

## 6. What to look for

- **Judge sub-scores** (grammar, creativity, consistency, meaningfulness,
  plot — Table 3 style): does one architecture reading consistently score
  better, or is it a mixed bag / within-noise at a single seed each? A
  large, consistent gap across all three sizes is real evidence for
  preferring that reading; a small or inconsistent gap suggests the
  ambiguity doesn't matter much in practice at this scale.
- **Parameter count**, per size, per strategy — report alongside judge
  scores every time (see §2). If `conv_on_image` wins on judge scores
  *and* uses fewer parameters, that's a genuinely strong, reproducible
  finding (efficiency + quality, not a tradeoff). If it wins only by
  using more parameters, that's a much weaker claim.
- **ROUGE-1** (vs. reference captions): a memorization/diversity check,
  not a quality signal — treat with real caution, especially given the
  earlier project's finding that ROUGE-1 trended opposite to judge
  quality as description length grew (longer/different output
  mechanically overlaps reference n-grams more, independent of actual
  quality). Don't let ROUGE decide a "which architecture is better"
  question on its own.
- **Consistency across sizes**: does whichever strategy wins at `mini`
  also win at `base` and `large`? If the winner flips with scale, that's
  itself a finding worth reporting, not noise to average away.
- **Single seed, single teacher=grader model — real caveats, not
  disclaimers to skip.** This is one run each per size/strategy
  combination, not a multi-seed average, so don't over-read a small gap.
  Teacher and grader being the same model (`llama3:8b`) carries a known
  self-preference risk in LLM-judge setups generally — worth keeping in
  mind when interpreting scores, even though resolving that risk is a
  separate experiment (not this one).

---

## 7. Definition of done

- All 6 checkpoints trained (2 architectures × 3 sizes), all 6 evaluated
  against the same 100-image held-out set.
- Parameter counts recorded for all 6.
- Three size-matched comparisons run and read (§5–6).
- A short written conclusion: which reading (if either) should be
  preferred as "the" replication going forward, with the actual numbers
  and caveats attached — not just a one-line verdict.
- `DESIGN_DECISIONS.md` in the new repo documents every filled-in gap
  (conv hyperparameters, epoch/batch choices, seed policy, held-out size
  and why it deviates from the paper's 25) so this is reproducible and
  the reasoning survives past this one session.
