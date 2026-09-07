# Conclusion: resolving the visual encoder ambiguity

`EXPERIMENT_GUIDE_encoder_ambiguity.md` §7 definition of done, closed out
here. All 6 checkpoints (2 encoder readings × 3 sizes) are trained and
evaluated against the same 100-image held-out set, same seed, same
tokenizer, same epoch/batch budget — the only thing that differs between
a `conv_on_patches` run and its `conv_on_image` counterpart at a given
size is the encoder itself. Full per-run numbers: `DESIGN_DECISIONS.md`
§11 (raw scores) and the three `compare_results.py` runs below (the
authoritative side-by-side).

## The numbers

| size  | conv_on_patches TOTAL | conv_on_image TOTAL | winner          | margin |
|-------|------------------------|----------------------|-----------------|--------|
| mini  | 18.51                  | 17.96                | conv_on_patches | +0.55  |
| base  | 19.27                  | 18.80                | conv_on_patches | +0.47  |
| large | 17.73                  | 17.98                | conv_on_image   | +0.25  |

Params, every size, both strategies (`conv_on_image` always smaller — a
side effect of a whole-image CNN stem vs. a per-patch conv stack at this
config, not something either reading was tuned to achieve):

| size  | conv_on_patches | conv_on_image | conv_on_image is smaller by |
|-------|-----------------|---------------|------------------------------|
| mini  | 5,691,664       | 5,107,088     | 10.3%                        |
| base  | 15,377,728      | 14,363,072    | 6.6%                         |
| large | 25,596,416      | 24,581,760    | 4.0%                         |

## Dimension-level pattern (more informative than TOTAL alone)

- **Consistency**: `conv_on_patches` wins at all three sizes (mini +0.51,
  base +0.18, large +0.16) — the one dimension where the direction holds
  across every scale tested, largest at `mini`.
- **Plot**: `conv_on_image` wins or ties at all three sizes (mini +0.27,
  base +0.07, large +0.02) — the mirror case, `conv_on_image`'s most
  reliable edge, though the gap shrinks to near-zero by `large`.
- **Grammar / creativity / meaningfulness**: mixed — `conv_on_patches`
  ahead at `mini`/`base`, `conv_on_image` ahead or roughly tied at
  `large`. These three dimensions are what actually drive the TOTAL
  flip at `large`: `conv_on_patches`'s `consistency` edge there (+0.16)
  is outweighed by `conv_on_image`'s combined edge on grammar (+0.10),
  creativity (+0.22), and meaningfulness (+0.07).

ROUGE-1 is *not* used to break any tie here (per §6's own caveat) —
worth noting `conv_on_image` has the higher ROUGE-1 at all three sizes
despite frequently losing on judge TOTAL, the same
higher-ROUGE-doesn't-mean-higher-quality pattern the earlier project saw.

## Verdict

**Prefer `conv_on_patches` (the paper's Section 2.2.1 text reading, not
the Figure 4 reading) as the default replication going forward, but treat
this as a modest, size-dependent preference rather than a decisive win.**

Reasoning: `conv_on_patches` wins TOTAL at 2 of 3 sizes, by larger margins
(+0.55, +0.47) than `conv_on_image`'s win at the third (+0.25) — and its
one dimension-level win (`consistency`) is the single most scale-robust
effect in the whole comparison, holding at every size tested. `conv_on_image`
is smaller everywhere and does win outright at `large` (better score, fewer
params — the "genuinely strong" case §6 calls out) — that is a real point in
its favor at that specific size, not one to explain away, and it means
"conv_on_patches, unconditionally" is too strong a claim.

## Caveats (load-bearing, not disclaimers)

- **Single seed, single run per (size, strategy) cell.** None of the
  margins above (0.16-0.55 points on a 25-point scale) have been checked
  against run-to-run variance. A gap this size could plausibly close or
  reverse under a different seed — this conclusion describes what
  happened in these 6 runs, not a statistically established effect.
- **Teacher = grader model (`llama3:8b` for both caption generation and
  judging)** carries a known self-preference risk in LLM-judge setups.
  Not resolved here (out of scope per the guide) — worth flagging in any
  external write-up of this result.
- **`large`'s checkpoints for both strategies were trained past their own
  best-val point** (`DESIGN_DECISIONS.md` §8: `val_loss` rises from
  epoch 7 onward for both `large/conv_on_patches` and
  `large/conv_on_image`, by design — the fixed 20-epoch budget was kept
  identical across strategies rather than early-stopped per run). The
  `large` comparison above is therefore a comparison of two
  similarly-overfit checkpoints, not each architecture's best achievable
  quality — a fair comparison *between the two strategies*, but not
  necessarily representative of `large`'s ceiling on this task.
- **The effect is not consistent in direction across scale** (`mini`/`base`
  favor `conv_on_patches`, `large` favors `conv_on_image`) — per §6, this
  is itself a finding, not noise to average away. It means the paper's
  ambiguity is not fully resolved by "one reading is just better" — the
  answer depends on model scale, at least at the three sizes tested here.

## Raw comparison output

The three `experiments/encoder_ambiguity/compare_results.py` runs behind
this conclusion (verbatim, `--labels conv_on_patches conv_on_image`):

```
=== size: mini ===
dimension       conv_on_patches  conv_on_image
grammar         2.91             2.70
creativity      2.91             2.84
consistency     4.34             3.83
meaningfulness  4.54             4.51
plot            3.81             4.08
TOTAL           18.51            17.96
Params: conv_on_patches=5,691,664  conv_on_image=5,107,088
ROUGE-1: conv_on_patches=0.5979  conv_on_image=0.6046

=== size: base ===
dimension       conv_on_patches  conv_on_image
grammar         3.07             2.87
creativity      3.12             2.79
consistency     4.28             4.10
meaningfulness  4.81             4.98
plot            3.99             4.06
TOTAL           19.27            18.80
Params: conv_on_patches=15,377,728  conv_on_image=14,363,072
ROUGE-1: conv_on_patches=0.6029  conv_on_image=0.6065

=== size: large ===
dimension       conv_on_patches  conv_on_image
grammar         2.64             2.74
creativity      2.76             2.98
consistency     4.17             4.01
meaningfulness  4.53             4.60
plot            3.63             3.65
TOTAL           17.73            17.98
Params: conv_on_patches=25,596,416  conv_on_image=24,581,760
ROUGE-1: conv_on_patches=0.6034  conv_on_image=0.6165
```

0/100 judge parse failures on all 6 runs — no data quality issue behind
any of the above.
