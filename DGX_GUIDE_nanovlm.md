# nanovlm — DGX setup & operating guide

Project-specific companion to the general Slurm/Ollama reference (carry
`DGX_GUIDE.md` from the old project forward as-is — nothing in it was
project-specific, it's cluster facts: MIG slices hanging on Ollama jobs,
the ambient-daemon-is-CPU-bound trap, `PYTHONUNBUFFERED=1`, etc.). This
doc is the new project's own paths, scripts, and conventions, written
clean rather than inherited from `microvlm-claude-version`.

Give this + `EXPERIMENT_GUIDE_encoder_ambiguity.md` to the Cowork session
that will actually write the code.

---

## 0. What's carried forward vs. rebuilt clean

**Carried forward (confirmed facts about this specific cluster, not old
project code):**
- Account `cs26d002`, compute node `dgx1` (no internet), login node (has
  internet), GPU is one A100-SXM4-40GB.
- Queues: `shortq` (~10 min, diagnostics), `mediumq` (~12h), `longq` (48h
  cap, main queue).
- `dgx1` allows exactly **one running job at a time** for this account
  (`AssocMaxJobsLimit`) — confirmed previously via
  `sacctmgr show assoc where user=cs26d002`.
- Conda env pattern:
  ```bash
  . /etc/profile.d/modules.sh
  module load anaconda/2023.03-1
  eval "$(conda shell.bash hook)"
  conda activate dgx-research-test
  ```
  (Same env, `/scratch/cs26d002/envs/dgx-research-test`, PyTorch
  2.6.0+cu118 — reuse it; no reason to rebuild.)
- Ollama shared model store: `/usr/share/ollama/.ollama/models`,
  world-readable, confirmed to have `llama3:8b`, `llama3:70b`,
  `llama2:13b`, `deepseek-r1:latest`, `nomic-embed-text`.
- The system `ollama` binary has no CUDA backend — a GPU-capable build
  needs fetching once via a `setup_ollama_gpu.sh`-equivalent script (see
  §3).
- Ambient Ollama daemon (port 11434) is CPU-bound and not part of any
  Slurm job — don't use it for real runs. Job-private server on port
  11435 is the pattern.
- MIG slices (`a100_1g.5gb`, `a100_3g.20gb`) are confirmed to **hang**
  specifically for Ollama-backed jobs (GPU init never completes). Use
  the full `--gres=gpu:a100:1` for `generate`/`evaluate`. Untested for
  pure-PyTorch `train` jobs — full GPU is the safe default there too
  unless you deliberately want to test a MIG slice on `shortq` first.

**Rebuilt clean (new this time):**
- **Git instead of zip+scp.** Repo: `git@github.com:research-pjy/nanovlm.git`.
  Local clone: `/home/jayanth/mainquests/nanovlm`. On the DGX, clone once
  on the **login node** (has internet + needs your SSH key added there),
  then `git pull` for every update — no more manual zip/scp/unzip.
- **New project folder name**: `nanovlm`, not `microvlm-claude-version`.
- **Repo path on DGX**: `/scratch/cs26d002/repos/nanovlm`.
- **`.gitignore`** excludes `data/`, `checkpoints/`, `results/`, `logs/`
  from commit 1 — same reason the old zips excluded them (generated
  output shouldn't round-trip through version control or get clobbered
  by a pull), but automatic instead of a manual exclude-list per zip.
- **Teacher/grader model**: `llama3:8b` (deliberately *not* `llama3:70b` —
  see §3 for why).
- **One experiment only right now**: resolving the paper's own visual
  encoder ambiguity (`conv_on_patches` vs `conv_on_image`). No sprawl of
  ten experiment folders this time — build this one, understand it fully,
  then decide what's next.

---

## 1. One-time DGX setup

```bash
# On the DGX LOGIN node (has internet)
mkdir -p /scratch/cs26d002/repos
cd /scratch/cs26d002/repos
git clone git@github.com:research-pjy/nanovlm.git
cd nanovlm
```

If SSH auth to GitHub isn't already set up on the login node, either add
a new deploy key/personal SSH key there, or clone via HTTPS with a
personal access token instead — either works, pick whichever you already
have credentials for.

Every future code update, on the login node:

```bash
cd /scratch/cs26d002/repos/nanovlm
git pull
```

`dgx1` (compute) never needs its own clone or its own internet access —
it just reads whatever's already in `/scratch/cs26d002/repos/nanovlm`
after the login node pulls.

### Conda env (reused, not rebuilt)

```bash
. /etc/profile.d/modules.sh
module load anaconda/2023.03-1
eval "$(conda shell.bash hook)"
conda activate /scratch/cs26d002/envs/dgx-research-test
```

If this env doesn't exist anymore for some reason, recreate from the
project's `environment.yml` (Cowork will write this) at that same path,
then `pip install -e .`.

### COCO images (needs the bigger set this time)

The old setup only downloaded 5000 `val2017` images. **28K pairs means
pulling from `train2017` too** (val2017 alone doesn't have enough
images). Selective, not the full 18GB zip — see
`EXPERIMENT_GUIDE_encoder_ambiguity.md` §1 for the exact script spec
(fetch annotations, pick 28K image IDs with a fixed seed, download only
those files, skip-existing so it's resumable).

```bash
# On the LOGIN node (needs internet)
python scripts/download_coco.py \
    --splits train2017 val2017 \
    --num-images 28000 \
    --seed 42 \
    --annotations-dir /scratch/cs26d002/datasets/coco/annotations \
    --images-dir /scratch/cs26d002/datasets/coco/images
```

(Adjust flags to match whatever Cowork actually implements — the point
is: run it on the login node, once, into the shared `datasets/` folder,
outside the project repo, so it's never re-downloaded per run.)

### GPU-capable Ollama build

```bash
# On the LOGIN node (needs internet), one-time
bash dgx/setup_ollama_gpu.sh
```

Downloads the official Ollama release (bundles CUDA backend libs) to
`/scratch/cs26d002/software/ollama-gpu/`. All `dgx/*.sbatch` scripts that
touch Ollama should check for this binary on `PATH` first and fall back
to the slow CPU one with a loud warning if it's missing.

---

## 2. Storage layout

```
/scratch/cs26d002/
  repos/nanovlm/                    <- this project (git-managed)
    data/processed/nanovlm_28k/     <- generated JSONL (gitignored)
    checkpoints/                     <- trained models (gitignored)
    results/                         <- eval JSON (gitignored)
    logs/                            <- sbatch stdout/stderr (gitignored)
    dgx/                             <- sbatch scripts
  datasets/coco/                    <- shared, outside the project, reused
                                        across any future project
  envs/dgx-research-test/           <- reused conda env
  software/ollama-gpu/              <- reused GPU-capable Ollama build
```

Rule of thumb unchanged from before: if regenerating it means
re-downloading from the internet, it lives in `datasets/`; if
regenerating it means re-running this project's own scripts, it lives
inside the project folder.

---

## 3. Teacher/grader model: `llama3:8b`, not `llama3:70b`

Considered `llama3:70b` (also in the shared store) but reverted to
`llama3:8b` for one concrete reason: `llama3:8b` has real measured
numbers on this exact GPU (~34s/call steady-state once GPU-offloaded,
confirmed in earlier work on this cluster), while `llama3:70b` has zero
benchmark history here and, on a single 40GB A100, may not comfortably
fit alongside everything else even before considering latency. At 28K
sequential calls, even `llama3:8b`'s known ~34s/call works out to
**~265 hours (~11 days) of pure generation time** — using an unbenchmarked,
likely-much-slower 70B model here would be a bad trade with no upside
established. If you want to explore `llama3:70b` later as its own
teacher-comparison experiment, benchmark it standalone first (a `shortq`
diagnostic, one real call, timed) before ever pointing a real run at it.

`llama2:13b` is the other technically-"smaller-than-70b" option in the
store but is untested here too — default to `llama3:8b` unless there's a
specific reason to branch.

---

## 4. The five pipeline stages — all manually triggered

**Nothing auto-chains between stages.** You run one `sbatch`, check the
result, decide whether to run the next. The only internal chaining is
*inside* stage 1 (`generate`), purely because 265h of sequential Ollama
calls can't fit in one 48h `longq` job — that chaining is invisible to
you as "the next stage running itself"; it's the same stage continuing
because it structurally has to.

| # | Stage | Trigger | sbatch script | Notes |
|---|---|---|---|---|
| 1 | `generate` | You run once | `dgx/generate.sbatch` | Self-chains ~6x internally until 28K captions exist. Resumes from wherever the incremental JSONL left off. |
| 2 | `train_conv_on_patches` × 3 | You run 3x (mini, base, large) | `dgx/train.sbatch` | One job per size. All three before moving to eval. |
| 3 | `eval_conv_on_patches` × 3 | You run 3x | `dgx/evaluate.sbatch` | All three before moving to `conv_on_image`. |
| 4 | `train_conv_on_image` × 3 | You run 3x | `dgx/train.sbatch` | Same script, different `--patch-embed-strategy` flag. |
| 5 | `eval_conv_on_image` × 3 | You run 3x | `dgx/evaluate.sbatch` | |

**13 total manual submissions.** No interleaving between architectures —
`conv_on_patches` fully trained *and* evaluated (all 3 sizes) before
`conv_on_image` starts. Within an architecture, train all 3 sizes first,
then evaluate all 3 — not train→eval→train→eval per size (that was
considered and deliberately rejected: you want the full picture per
architecture before reviewing anything).

### Stage 1 — generate

```bash
cd /scratch/cs26d002/repos/nanovlm
mkdir -p logs
sbatch dgx/generate.sbatch
squeue -u $USER
tail -f logs/nanovlm-generate-<jobid>.out
```

Self-chains until `data/processed/nanovlm_28k/{train,val,eval_holdout}.jsonl`
are complete. Check `squeue -u $USER` occasionally over the following
days — you'll see at most one `generate` job at a time, each new one
appearing automatically as the previous finishes. If a chain segment
fails, it stops (doesn't silently continue), same pattern as the old
project's length-sweep: check the failed segment's log, fix, resubmit
manually from where it left off (the script should support a `--resume`
flag that skips already-written records).

### Stages 2–5 — train / evaluate, one size at a time

```bash
cd /scratch/cs26d002/repos/nanovlm

# Example: train conv_on_patches, base size
sbatch --export=ALL,SIZE=base,STRATEGY=conv_on_patches dgx/train.sbatch
squeue -u $USER
tail -f logs/nanovlm-train-<jobid>.out

# ... check it finished cleanly, then when ready:
sbatch --export=ALL,SIZE=base,STRATEGY=conv_on_patches dgx/evaluate.sbatch
```

Repeat for `mini`/`large`, then repeat the whole block with
`STRATEGY=conv_on_image` once all three `conv_on_patches` evals are done
and reviewed.

---

## 5. Naming convention

```
data/processed/nanovlm_28k/{train,val,eval_holdout}.jsonl   <- shared, generated once
checkpoints/nanovlm_<size>_<strategy>/final.pt                <- e.g. nanovlm_base_conv_on_image
results/nanovlm_<size>_<strategy>_eval.json
logs/nanovlm-<stage>-<jobid>.out / .err
```

`<size>` ∈ `{mini, base, large}`, `<strategy>` ∈
`{conv_on_patches, conv_on_image}`. Six checkpoint/result pairs total.

**Job names deliberately don't say "nanovlm"** in `--job-name` (carried
forward from the old project's convention) — `squeue` is visible
cluster-wide, keep job names generic (`gen`, `train-base-pat`,
`eval-large-img`, etc.) even though file paths inside your own project
folder can say whatever you want.

---

## 6. Before submitting the real 28K generate job

- [ ] Confirm images are actually downloaded at the scale requested — add
      a preflight check to `generate.sbatch` that counts files on disk
      against what the job is about to request, and aborts loudly if
      short, rather than silently generating fewer than 28K (this bit the
      old project once at a smaller scale — see old `DGX.md` §9's
      preflight-check note; don't skip it this time just because it
      wasn't literally re-experienced yet in this repo).
- [ ] Run `dgx/test_ollama_gpu.sbatch` fresh once, confirm `llama3:8b`
      loads via the private GPU-capable server, before trusting the real
      28K job to use it correctly.
- [ ] Confirm `--mem` and `--time` on `generate.sbatch` are sized from a
      real single-call timing check on *this* cluster today, not copied
      from old numbers — cluster conditions can drift.
- [ ] Confirm the incremental-write + `--resume` behavior actually works
      by killing a `shortq`-scale test run partway through and restarting
      it, before trusting it across a multi-day, multi-job chain.

---

## 7. Open items to settle in the Cowork build session

Not resolved in this doc — hand these to Cowork explicitly rather than
letting it guess:

- Exact conv kernel/stride/padding for both patch-embedding strategies
  (a documented default exists from the earlier project's
  `DESIGN_DECISIONS.md` — reuse it or reconsider it, but decide
  explicitly and write it down again in this repo's own
  `DESIGN_DECISIONS.md`, don't silently inherit).
- Epoch count / batch size per size (old runs used 20 epochs / batch 64
  for `base` — decide whether that carries over as-is for all three
  sizes or needs adjusting for `mini`/`large`).
- Exact `--mem`/`--time` values for each sbatch script — size from a real
  `shortq` measurement on this repo's actual code, not inherited blindly.
- Random seed policy — one fixed seed (e.g. 42) reused across data split,
  tokenizer training, and both architectures' model init, so the only
  variable between the two architecture runs is genuinely just the
  encoder.
