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

**Update — first `conv_on_patches` results in hand for all three sizes.**
Actual behavior was the opposite of the "large still improving at epoch
20" case this section flagged as worth watching for: `large` instead
*overfits* within the 20-epoch budget — `val_loss` bottoms at epoch 7
(2.4556) and rises every epoch after, ending 2.6248 at epoch 20, while
`train_loss` keeps falling the whole time (→1.7406). `base` shows a much
milder version (val bottoms epoch 12-13, drifts up slightly by 20).
`mini` barely shows it. `scripts/train.py` only ever saves the
final-epoch checkpoint, never a best-val one, so the `large` checkpoint
on disk is meaningfully worse than that architecture's own best point.

**Decision: keep the final-epoch checkpoint for every size, do not add
best-val checkpointing.** No retraining, no `train.py` change. This
follows directly from this section's own stated goal — same epoch/batch
budget applied identically regardless of outcome, so the strategy
comparison (`conv_on_patches` vs `conv_on_image`) is never confounded by
a training-protocol difference — and it generalizes cleanly:
`conv_on_image`'s `large` run gets the exact same fixed-budget treatment,
whatever it does to that architecture's own overfitting curve. `large`'s
late-epoch overfitting is now a documented, reportable finding of the
study rather than something corrected for after the fact.

**Update — `conv_on_image` trained for all three sizes.** Same overfitting
pattern as `conv_on_patches`: `large` again rises in `val_loss` from
epoch 7 (2.4648) onward while `train_loss` keeps falling, applying the
same keep-final.pt policy above. One additional wrinkle unique to this
run: `large/conv_on_image` had a one-epoch `val_loss` spike at epoch 12
(3.0185, versus ~2.47-2.49 the epochs immediately before and 2.5170 the
epoch right after) — a transient blip, not a sustained divergence
(training continued normally afterward, following the same rising trend
as before the spike). Most likely an unusually hard/large batch rather
than a real instability, since it self-corrected in one epoch; noted here
in case it recurs on a future run and starts looking like a pattern
rather than noise. `mini`/`base` conv_on_image show the same mild-to-none
overfitting shape as their conv_on_patches counterparts.

---

## 9. `--mem` / `--time` sbatch values, and which partition is actually usable

**`--mem`/`--time` not filled in with real numbers** —
`EXPERIMENT_GUIDE_encoder_ambiguity.md` §7 / `DGX_GUIDE_nanovlm.md` §6 both
require these to come from a real timing measurement on *this* cluster,
*this* repo's actual code, not inherited or guessed. The `dgx/*.sbatch`
scripts in this repo carry placeholder values — filling these in requires
actually running on `dgx1`, which this build session cannot do. Do not
raise these past the placeholder without running `dgx/test_ollama_gpu.sbatch`
first; do not trust the placeholder for a real submission either.

**Update — first real `train.sbatch` run (`SIZE=mini,STRATEGY=conv_on_patches`,
job 46413) completed successfully, giving real numbers for the first
time.** 20/20 epochs, ~130.6s/epoch, ~44 minutes total wall time —
loss went 3.94→2.37 (train) and 3.25→2.53 (val), a normal-looking
monotonic decrease with no divergence or NaNs. `--time=11:30:00` is
enormously oversized for `mini` (44min actual vs 11.5h budgeted); left
as-is rather than tightened, since the margin costs nothing (this
account's jobs are already serialized on `dgx1` one-at-a-time, so a
generous `--time` doesn't block other work) and `base`/`large` are
untimed. `--mem=16G` was sufficient — no OOM, no `AssocGrpMemLimit`
block on this submission. Per-epoch time here is realistically
dominated by the image-loading/decode side of the `DataLoader`
(`num_workers=4`), not model compute — `mini`'s 5.7M total params is
tiny for an A100 — so `base` (n_blks=3, n_embd=128) and `large`
(n_blks=5, n_embd=192, Table 1) are expected to land in a similar
per-epoch ballpark rather than scaling badly, but that is still an
expectation, not yet a measurement: confirm with their own first runs
before assuming it holds.

**Partition is `longq` everywhere, not `mediumq`/`shortq` as
`DGX_GUIDE_nanovlm.md` §0 lists them — and `--qos=` must be set
explicitly, `--partition` alone is not enough.** `mediumq`, then `shortq`,
then even `longq` with no `--qos=` set all failed with
`sbatch: error: Batch job submission failed: Invalid qos specification`
for this account when actually submitted. Running
`sacctmgr show assoc where user=cs26d002 format=account,partition,qos`
gave the authoritative answer:

```
   Account  Partition                  QOS
---------- ---------- --------------------
   student                    longq,shortq
```

This account's Slurm account name is `student` (not `cs26d002`), it has no
partition restriction, and its allowed QOS set is exactly `{longq,
shortq}` with no usable default QOS — so any job that doesn't explicitly
request `--qos=` gets assigned a default QOS outside that set and is
rejected, *regardless of which `--partition` is named*. `mediumq` failed
because it isn't in the allowed set at all (as either partition or QOS);
`shortq` and the first `longq` attempt failed purely because `--qos=` was
never set. Every `dgx/*.sbatch` script now sets both
`--partition=longq` and `--qos=longq` together — matching QOS to
partition — regardless of how short the job's own `--time` is; `--time`
only caps how long that job may run, not how long it has to wait. If a
job genuinely wants the short queue instead, `--partition=shortq
--qos=shortq` (both set, matching) should work by the same logic; this
hasn't been submitted yet, so treat it as unconfirmed until it is.

**`--mem=32G` on `generate.sbatch` caused a second, separate blocker:
`squeue` showed the job PD with reason `AssocGrpMemLimit`** — the
`student` account's group memory quota (shared across every user on that
account, per the same `sacctmgr` output above — other pending jobs in
`squeue` belonged to different usernames, meaning many users share this
one account) was already spoken for by other users' jobs, so this job
couldn't start regardless of GPU availability. Lowered `generate.sbatch`
to `--mem=16G` — not a blind reduction, it matches `test_ollama_gpu.sbatch`,
which already succeeded at `--mem=16G` running the same private-Ollama-
server, one-call-at-a-time workload. If a job is still PD on
`AssocGrpMemLimit` after this, that means other users' jobs are eating the
shared quota, not this job's own request, and lowering `--mem` further
won't fix it — confirm the account's actual cap first with
`sacctmgr show assoc where account=student format=account,user,grptres,maxtresperjob`
rather than guessing another number.

**`train.sbatch` hit the identical `AssocGrpMemLimit` block at `--mem=32G`
on its very first real submission (`SIZE=mini,STRATEGY=conv_on_patches`,
job 46412).** Lowered to `--mem=16G`, same fix as `generate.sbatch` above.
This one is a slightly different workload (real PyTorch training via
`scripts/train.py`, not an Ollama server), but the model and optimizer
state live on the GPU, not host RAM — host RAM here is only the
DataLoader pipeline (`--num-workers 4`) and Python/CUDA overhead — so 16G
should still be generous for all three sizes. If `SIZE=large` specifically
OOMs on host RAM (check `sacct -j <jobid> --format=JobID,MaxRSS,State`
after it finishes or fails), raise `--mem` for that one job rather than
reflexively raising it for all three.

---

## 10. Storage / naming

No deviation from `DGX_GUIDE_nanovlm.md` §2 and §5 — reused as specified:
`data/processed/nanovlm_28k/{train,val,eval_holdout}.jsonl`,
`checkpoints/nanovlm_<size>_<strategy>/final.pt`,
`results/nanovlm_<size>_<strategy>_eval.json`,
`logs/nanovlm-<stage>-<jobid>.out/.err`, generic Slurm job names (not
containing "nanovlm").

---

## 11. Results log (running)

Raw numbers as each `evaluate.sbatch` run completes, so the final
`compare_results.py` step and the paper's conclusion aren't the first
place these are written down. Full detail lives in
`results/nanovlm_<size>_<strategy>_eval.json`; this is just a running
summary table.

| size  | strategy         | grammar | creativity | consistency | meaningfulness | plot | avg_total | rouge1 |
|-------|------------------|---------|------------|-------------|-----------------|------|-----------|--------|
| mini  | conv_on_patches  | 2.91    | 2.91       | 4.34        | 4.54            | 3.81 | 18.51     | 0.5979 |
| base  | conv_on_patches  | 3.07    | 3.12       | 4.28        | 4.81            | 3.99 | 19.27     | 0.6029 |
| large | conv_on_patches  | 2.64    | 2.76       | 4.17        | 4.53            | 3.63 | 17.73     | 0.6034 |
| mini  | conv_on_image    | 2.70    | 2.84       | 3.83        | 4.51            | 4.08 | 17.96     | 0.6046 |
| base  | conv_on_image    | 2.87    | 2.79       | 4.10        | 4.98            | 4.06 | 18.80     | 0.6065 |
| large | conv_on_image    | 2.74    | 2.98       | 4.01        | 4.60            | 3.65 | 17.98     | 0.6165 |

All 6 runs (both strategies x 3 sizes) are complete as of this update —
this table now has every number `experiments/encoder_ambiguity/
compare_results.py` needs; its own per-size runs are the authoritative
comparison, this is just a preview.

`conv_on_patches` observation: `base` scores highest on `avg_total`
(19.27), not `large` (17.73, the lowest of the three) — consistent with
§8's overfitting finding: `large`'s held-out judge quality actually
degrades relative to `base` despite having ~1.7x the parameters, which
tracks with its `val_loss` having risen from epoch 7 onward at the
checkpoint actually being evaluated (`final.pt`, epoch 20). `mini` sits
between the two on `avg_total` despite being smallest, so this isn't a
clean monotonic size effect — more evidence for treating `large`'s result
as budget-limited (§8) rather than reading `avg_total` as tracking
parameter count directly.

Cross-strategy preview (`conv_on_patches` `avg_total` minus
`conv_on_image` `avg_total`, per size): `mini` +0.55, `base` +0.47,
`large` −0.25. `conv_on_patches` wins at `mini` and `base`;
`conv_on_image` edges ahead at `large`, though by less than either
`mini`/`base` gap and inside the range that could plausibly be single-
seed, single-judge-model noise (`EXPERIMENT_GUIDE_encoder_ambiguity.md`
§6's own caveat) rather than a real crossover. Worth stating as "the
effect direction is not fully consistent across sizes" in the writeup
rather than as a confirmed size-dependent reversal — `compare_results.py`
is the tool for the real per-size read, this is only a heads-up on what
it's likely to show.
