# autoresearch-gaudi

Autonomous LM training research framework for Intel Gaudi 3, ported from [autoresearch-cuda](https://github.com/elementalcollision/autoresearch-cuda).

An autonomous agent (Claude) iteratively modifies hyperparameters, trains a small language model on a Gaudi 3 HPU, evaluates val_bpb (validation bits per byte), and keeps improvements -- fully unattended.

## Hardware

- **Intel Gaudi 3** on IBM Cloud (`gx3d-160x1792x8gaudi3`)
- 8 accelerators per instance, 128 GB HBM2e each
- ~1835 TFLOPS bf16 per device
- 64 Tensor Processor Cores (TPCs) per device

## Quick Start

### 1. Set up IBM Cloud instance

```bash
# SSH into your Gaudi 3 instance, then:
bash setup_ibm_cloud.sh
```

### 2. Build and run with Docker

```bash
# Build the container
docker compose build

# Verify HPU access
docker compose run verify

# Benchmark HPU performance
docker compose run --rm train python scripts/benchmark_hpu.py

# Prepare data (download + tokenize)
docker compose run prepare

# Run a single training experiment
docker compose run train

# Run the autonomous agent (headless, no TUI)
docker compose run agent
```

### 3. Or run directly (inside the Habana container)

```bash
# Install dependencies
pip install numpy pyarrow requests rustbpe tiktoken anthropic textual

# Prepare data
python prepare.py

# Train
python -u train_gaudi.py

# Run headless agent
python -m tui.headless --tag mar22 --max 80
```

## Architecture

| Component | File | Description |
|-----------|------|-------------|
| Training | `train_gaudi.py` | GPT model + training loop (single HPU) |
| Data | `prepare.py` | Download, tokenize, dataloader, evaluation |
| Optimizer | `backends/muon_gaudi.py` | MuonAdamW with Newton-Schulz orthogonalization |
| Hardware | `backends/__init__.py` | Gaudi 3 detection via hl-smi |
| Agent | `tui/orchestrator.py` | Autonomous experiment loop |
| TUI | `dashboard.py` | Interactive terminal dashboard |
| Headless | `tui/headless.py` | Agent without TUI (for unattended runs) |
| Multi-dataset | `run_suite.py` | Run across multiple datasets |

## Model

Small GPT with:
- ResFormer value embeddings
- Sliding window + full attention (SSSL pattern)
- Rotary position embeddings
- RMS normalization
- MuonAdamW optimizer (Newton-Schulz orthogonalization + AdamW for embeddings)

Default hyperparameters scaled for Gaudi 3:

| Parameter | Value |
|-----------|-------|
| DEPTH | 16 |
| DEVICE_BATCH_SIZE | 64 |
| TOTAL_BATCH_SIZE | 262144 tokens |
| ASPECT_RATIO | 64 |
| TIME_BUDGET | 300s |

## Modes

### Single training run
```bash
python -u train_gaudi.py
```

### Interactive dashboard (TUI)
```bash
python dashboard.py                    # Single run
python dashboard.py --agent            # Autonomous agent
python dashboard.py --agent --max 50   # Limit experiments
python dashboard.py --watch            # Monitor only
```

### Headless agent (recommended for unattended runs)
```bash
python -m tui.headless --tag mar22 --max 80
```

### Multi-dataset suite
```bash
python run_suite.py --status           # Check status
python run_suite.py --dataset fineweb-edu --max-experiments 80
python run_suite.py                    # Run all datasets
```

## Credential Setup

The autonomous agent requires a Claude API key:

```bash
# Option 1: Environment variable
export ANTHROPIC_API_KEY=sk-ant-...

# Option 2: File-based store
python dashboard.py --setup-key

# Check current credential
python dashboard.py --check-key
```

## Docker

Based on `vault.habana.ai/gaudi-docker/1.23.0/ubuntu24.04/habanalabs/pytorch-installer-2.5.1:latest`.

**Important**: Do not `pip install torch` -- the Habana container ships a custom PyTorch build with HPU backend compiled in.

## Key Differences from Apple Silicon / CUDA Versions

| Aspect | CUDA | Apple Silicon | Gaudi 3 |
|--------|------|---------------|---------|
| Device | `torch.device("cuda")` | `torch.device("mps")` | `torch.device("hpu")` |
| Compile | `torch.compile()` | Eager mode | `torch.compile(backend="hpu_backend")` |
| Attention | FlashAttention-3 | SDPA | FusedSDPA / SDPA |
| Memory | NVIDIA HBM | Unified (shared) | 128 GB HBM2e |
| Monitoring | nvidia-smi | Activity Monitor | hl-smi |

## Files

```
.
├── train_gaudi.py          # Main training script (single HPU)
├── prepare.py              # Data pipeline
├── dashboard.py            # TUI entry point
├── monitor.py              # Progress monitor
├── run_suite.py            # Multi-dataset orchestration
├── convert_dataset.py      # Alternative dataset conversion
├── compare_datasets.py     # Cross-dataset analysis
├── program.md              # Agent instructions
├── Dockerfile
├── docker-compose.yml
├── setup_ibm_cloud.sh
├── pyproject.toml
├── backends/
│   ├── __init__.py         # Gaudi 3 hardware detection
│   └── muon_gaudi.py       # MuonAdamW optimizer
├── tui/
│   ├── app.py              # TUI dashboard
│   ├── credentials.py      # API key management
│   ├── hardware.py         # hl-smi hardware info
│   ├── headless.py         # Headless agent runner
│   ├── llm_backend.py      # Claude API backend
│   ├── orchestrator.py     # Experiment loop
│   ├── parser.py           # Output parser
│   ├── results.py          # Results TSV management
│   ├── widgets.py          # TUI widgets
│   └── styles.tcss         # TUI styles
├── scripts/
│   ├── verify_hpu.py       # HPU sanity check
│   └── benchmark_hpu.py    # Performance benchmarks
└── docs/
    └── evaluating-results.md
```
