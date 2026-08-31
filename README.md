# nanovlm

Resolving the [NanoVLMs paper](https://arxiv.org/abs/2502.07838)'s own
visual encoder ambiguity: Section 2.2.1's text describes convolution
applied *per patch*; Figure 4 reads as a *whole-image* CNN stem. Both are
implemented here, trained under identical conditions, and compared.

Read these three docs before touching anything — everything below assumes
them:

- **`DGX_GUIDE_nanovlm.md`** — cluster facts, storage layout, the
  five-stage pipeline, how to actually submit jobs.
- **`EXPERIMENT_GUIDE_encoder_ambiguity.md`** — the research design: what
  the ambiguity is, what's held constant between the two arms, how to read
  results.
- **`DESIGN_DECISIONS.md`** — every gap the paper leaves unfilled, decided
  once and written down: conv hyperparameters, attention head sizing,
  tokenizer choice, epoch/batch size, seed policy, held-out set size, and
  why this build uses one caption format instead of the paper's
  ShortDesc+LongDesc pair.

## Layout

```
nanovlm/                   # the model + data library (pip install -e .)
  config.py                 # Table 1 hyperparameters (mini/base/large) + fixed hparams
  model/                     # encoder (both strategies), connector, decoder, full model
  data/                       # tokenizer, dataset, prompts
  metrics.py                   # ROUGE-1 (diversity check, not a quality signal)
  ollama_client.py              # shared HTTP client for the job-private Ollama server
scripts/
  download_coco.py           # stage 0 — selective COCO download (login node)
  train_tokenizer.py          # stage 0 — train the shared tokenizer once
  generate_captions.py         # stage 1 — llama3:8b teacher captions
  train.py                      # stages 2/4 — train one (size, strategy) checkpoint
  evaluate.py                    # stages 3/5 — evaluate one checkpoint
experiments/encoder_ambiguity/
  compare_results.py         # stage 5 follow-up — N-way results comparison
dgx/                        # sbatch scripts + Ollama setup, see DGX_GUIDE_nanovlm.md
tests/                      # local sanity tests (shapes, tiny forward/backward, etc.)
data/ checkpoints/ results/ logs/   # gitignored — generated, never committed
```

## Quickstart (on the DGX login node, per DGX_GUIDE_nanovlm.md)

```bash
# one-time
bash dgx/setup_ollama_gpu.sh
python scripts/download_coco.py \
    --splits train2017 val2017 --num-images 28000 --seed 42 \
    --annotations-dir /scratch/cs26d002/datasets/coco/annotations \
    --images-dir /scratch/cs26d002/datasets/coco/images

# on the compute node (dgx1), via sbatch — see dgx/*.sbatch and
# DGX_GUIDE_nanovlm.md §4 for the full 13-submission sequence
sbatch dgx/generate.sbatch
python scripts/train_tokenizer.py \
    --train-jsonl data/processed/nanovlm_28k/train.jsonl \
    --out-dir data/processed/nanovlm_28k/tokenizer
sbatch --export=ALL,SIZE=base,STRATEGY=conv_on_patches dgx/train.sbatch
sbatch --export=ALL,SIZE=base,STRATEGY=conv_on_patches dgx/evaluate.sbatch
```

Compare results once all 6 checkpoints are trained and evaluated:

```bash
python experiments/encoder_ambiguity/compare_results.py \
    results/nanovlm_base_conv_on_patches_eval.json \
    results/nanovlm_base_conv_on_image_eval.json \
    --labels conv_on_patches conv_on_image
```

## Local dev / sanity checks

No DGX or GPU needed for these — they run on CPU with tiny synthetic data:

```bash
pip install -e ".[dev]"
pytest tests/
```

## Definition of done

Per `EXPERIMENT_GUIDE_encoder_ambiguity.md` §7: all 6 checkpoints trained
and evaluated against the same 100-image held-out set, parameter counts
recorded for all 6, three size-matched comparisons run, and a short written
conclusion — with actual numbers and caveats, not a one-line verdict —
on which encoder reading (if either) should be preferred going forward.
