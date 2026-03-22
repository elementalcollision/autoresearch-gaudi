# autoresearch (Intel Gaudi 3)

This is an experiment to have the LLM do its own research, ported to Intel Gaudi 3 HPU on IBM Cloud.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autoresearch/<tag>` must not already exist -- this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` -- repository context.
   - `prepare.py` -- fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train_gaudi.py` -- HPU backend training script. This is the file you edit.
   - `backends/` -- hardware detection, optimizers. Do not modify.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Platform: Intel Gaudi 3 (IBM Cloud)

This port runs on Intel Gaudi 3 HPUs via IBM Cloud (`gx3d-160x1792x8gaudi3` instances). Key characteristics:

- **8x Gaudi 3 accelerators** per instance, 128 GB HBM2e each (~1 TB total).
- **~1835 TFLOPS bf16** per device -- significantly more compute than consumer GPUs or Apple Silicon.
- **torch.compile(backend="hpu_backend")**: Preferred compilation mode. Avoids manual `htcore.mark_step()` placement.
- **FusedSDPA**: Habana's optimized attention kernel for full-context (is_causal=True) attention.
- **Docker-based**: All training runs inside the Habana Docker container.
- **hl-smi**: Hardware monitoring tool (equivalent of nvidia-smi).

### Running training

```bash
# Inside Docker container
python -u train_gaudi.py

# Or via docker compose
docker compose run train
```

### Key differences from CUDA/Apple Silicon

- Device: `torch.device("hpu")` (not "cuda" or "mps")
- Import: `import habana_frameworks.torch` must come before any torch operations
- Compilation: `torch.compile(model, backend="hpu_backend")` (not "inductor")
- Autocast: `torch.amp.autocast(device_type="hpu", dtype=torch.bfloat16)`
- No FlashAttention-3: Uses FusedSDPA or standard SDPA
- Multiprocessing: Must use `spawn` or `forkserver` (not `fork`)

## Experimentation

Each experiment runs on a single Gaudi 3 HPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup). You launch it simply as: `python -u train_gaudi.py`.

**What you CAN do:**
- Modify `train_gaudi.py` -- this is the file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants.
- Modify `backends/`. The optimizer and hardware detection code is shared infrastructure.
- Install new packages or add dependencies.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest val_bpb.** Since the time budget is fixed, you don't need to worry about training time -- it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Memory** is a soft constraint. Each Gaudi 3 has 128 GB HBM2e. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome -- that's a simplification win.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     8192.0
mfu_percent:      45.20
total_tokens_M:   125.0
num_steps:        250
num_params_M:     50.3
depth:            16
backend:          hpu
chip:             Gaudi 3
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 9 columns:

```
exp	description	val_bpb	peak_mem_gb	tok_sec	mfu	steps	status	notes
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar22`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune the training script (`train_gaudi.py`) with an experimental idea
3. git commit
4. Run the experiment: `python -u train_gaudi.py > run.log 2>&1` (redirect everything)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace
7. Record the results in the tsv (do not commit results.tsv)
8. If val_bpb improved (lower), keep the commit
9. If val_bpb is equal or worse, git reset back

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup overhead). If a run exceeds 10 minutes, kill it and treat it as a failure.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. The human might be asleep. You are autonomous. The loop runs until the human interrupts you, period.
