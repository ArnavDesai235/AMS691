# Backend and Cluster (Ollama on ai-cluster-slurm)

Owner: Arnav. Team decision: every LightRAG LLM and embedding call (extraction, merge summaries, C4 summaries, answers) runs on a local Ollama server on one H100. Corpus generation (`corpus_gen/`) stays on Claude; the dev corpus is already built and `corpus_gen/llm.py` has no Ollama backend.

Machine-readable record of everything below: `configs/cluster.yaml` (recorded 2026-10-19 with `scontrol`, `sinfo`, `sacctmgr`, `ollama show`, `nvidia-smi`).

## 1. Cluster

| Item | Value |
|---|---|
| Cluster | `ai-cluster-slurm` (login node `ai-slurm01`), Slurm 23.11.4 |
| Partition | `defq`: the only partition; 7-day limit |
| Account / QoS | `default` / `normal`; no per-user job or GPU limits |
| Node used | **`h100`**: 4 × NVIDIA H100 80 GB, 64 CPUs, 385 GB RAM |
| Storage | `/home` (NFS, 105 TB); no `$SCRATCH` |

| Node | GPUs | GPU model | CPUs | RAM |
|---|---|---|---|---|
| `h100` | 4 | H100 80 GB | 64 | 385 GB |
| `orion` | 4 | not recorded | 72 | 1.5 TB |
| `quadro1`, `quadro2` | 8 each | not recorded | 64 | 385 GB |
| `tesla1`, `tesla2` | 8 each | Tesla V100-SXM2 32 GB | 64 | 385 GB |

**GPU types are not labelled in Slurm** (`gres/gpu=N`, no type), so `--gres=gpu:1` alone can land on a 32 GB V100. That happened on the first test job. The 32B model needs about 67 GB, so every Ollama job must use `--nodelist=h100`. The job script does this, and exits if the GPU has under 70 GB. Driver 550.163.01, CUDA 12.4.

## 2. Models (`configs/rag.yaml`, `backend:`)

| Role | Ollama tag | ID | Size | Details |
|---|---|---|---|---|
| LLM (extraction, summaries, answers) | `qwen3:32b-q8_0` | `a46beca077e5` | 35 GB | qwen3, 32.8B params, Q8_0, context 40960 |
| Embedding | `bge-m3:latest` | `790764642607` | 1.2 GB | bert, 566.7M params, F16, context 8192, **1024 dims** |

Options sent on every LLM call, overriding the model's defaults:

| Option | Ours | Model default | Why |
|---|---|---|---|
| `num_ctx` | 32768 | Ollama's small default | A 1200-token chunk plus LightRAG's extraction prompt and examples overflows the default, and Ollama truncates it without an error, which drops entities and breaks exact-name merges. `rag_factory.py` refuses values under 8192. |
| `think` | false | true | Qwen3's reasoning text breaks LightRAG's extraction format. |
| `temperature` | 0 | 0.6 | Re-ingests and the cf sweep should build the same graph. |
| `seed` | 42 | none | Same reason. |

Embedding calls use `num_ctx 8192`. The same LLM is used for every stage: the hub summarizer decides how restricted text blends into merged descriptions, so the model is part of the experiment. Report `qwen3:32b-q8_0` with every result.

**GPU memory:** 35 GB of weights plus a KV cache of about 32 GB (4 parallel requests × 32k tokens × ~256 KB per token) comes to about 67 GB. That fits one H100 and does not fit a V100.

## 3. Software (conda env `AMS691`)

| Component | Version |
|---|---|
| Ollama server | 0.40.6 (`$CONDA_PREFIX/bin/ollama`, bundled `cuda_v12`, `cuda_v13`, `vulkan` libs) |
| Ollama Python client | 0.6.3 |
| lightrag-hku | 1.5.7 (pinned) |
| tiktoken / PyYAML / numpy | 0.14.0 / 6.0.3 / 2.5.4 |
| anthropic / wordfreq / english-words / pytest | 1.15.0 / 3.1.1 / 2.0.2 / 9.1.1 |
| Python | 3.15.0 as installed. The test suite is verified on 3.12; `environment.yml` pins 3.12 |

Env vars, set with `conda env config vars`: `OLLAMA_MODELS=/home/armdesai/ollama_models` (34 GB) and `TIKTOKEN_CACHE_DIR=$CONDA_PREFIX/share/tiktoken`, cached for compute nodes without internet.

## 4. One-time setup (login node)

```bash
conda env create -f environment.yml && conda activate AMS691

# Ollama server binary into the env (no sudo); the old .tgz URL now 404s
curl -fL https://ollama.com/download/ollama-linux-amd64.tar.zst | zstd -dc | tar -x -C "$CONDA_PREFIX"
ls $CONDA_PREFIX/lib/ollama          # must list cuda_v12 / cuda_v13, otherwise it runs on CPU

conda env config vars set OLLAMA_MODELS=$HOME/ollama_models TIKTOKEN_CACHE_DIR=$CONDA_PREFIX/share/tiktoken
conda deactivate && conda activate AMS691
mkdir -p "$OLLAMA_MODELS" "$TIKTOKEN_CACHE_DIR"
python -c "import tiktoken; tiktoken.encoding_for_model('gpt-4o-mini')"

# Pull models through a temporary CPU server (run inside tmux)
export OLLAMA_HOST=127.0.0.1:$(python -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1])')
ollama serve > ~/ollama_pull.log 2>&1 &
sleep 5 && ollama pull qwen3:32b-q8_0 && ollama pull bge-m3 && ollama list
kill %1 && unset OLLAMA_HOST

pytest -q tests
```

Teammates can point `OLLAMA_MODELS` at `/home/armdesai/ollama_models` (if readable) instead of pulling 36 GB again.

## 5. Running

```bash
sbatch scripts/run_ollama_job.sbatch                                     # ingest only
STEPS="ingest C1 C2 C3 C4 metrics" sbatch scripts/run_ollama_job.sbatch  # full pass
squeue -u $USER; tail -f logs/ams691-ollama-<jobid>.out
```

The job:
1. Pins `h100` and checks GPU memory.
2. Starts `ollama serve` on a free port, with 4 parallel requests to match `llm_max_async: 4`, the model kept loaded, and flash attention on.
3. Points a temporary copy of `rag.yaml` at that port.
4. Loads the LLM once and prints `ollama ps`. Check that PROCESSOR reads `100% GPU` before trusting latency numbers.
5. Runs `STEPS`.

The `h100` node is shared (state MIXED), so a job can wait in the queue when its GPUs are busy.

## 6. Before trusting results

After the first ingest, check `ingest_report.json`:
- the exact-canonical-name match rate;
- whether every cross-tier entity merged, meaning its source chunks come from both tiers;
- the realized hub regimes.

Low exact-name coverage means extraction, not enforcement, is the bottleneck.
