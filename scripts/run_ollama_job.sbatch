#!/usr/bin/env bash
#SBATCH --job-name=ams691-ollama
#SBATCH --partition=defq            # ai-cluster-slurm's only partition
#SBATCH --nodelist=h100             # REQUIRED: GRES is untyped, gpu:1 alone can land on a 32 GB V100
#SBATCH --gres=gpu:1                # one H100 80 GB (node has 4)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#
# Cluster: ai-cluster-slurm (login ai-slurm01). Values: configs/cluster.yaml. Setup: DESIGN_backend.md.
# Starts Ollama on the allocated H100, then runs the pipeline against it on the same node.
#   sbatch scripts/run_ollama_job.sbatch                       # ingest only
#   STEPS="C1 C2 C3 C4 metrics leakage" sbatch scripts/run_ollama_job.sbatch
#   (check, metrics and leakage need no GPU and can also run on the login node)
#   INGEST_ARGS="--retry-truncated --rounds 3" sbatch scripts/run_ollama_job.sbatch
#   STEPS=corpus sbatch scripts/run_ollama_job.sbatch            # generate configs/full.yaml's corpus locally
# Submit from the repo root. Env vars: ENV_NAME (AMS691), STEPS, RAG_CONFIG (configs/rag.yaml), INGEST_ARGS,
#   CORPUS_CONFIG (configs/full.yaml)
set -euo pipefail

ENV_NAME=${ENV_NAME:-AMS691}
STEPS=${STEPS:-ingest}
RAG_CONFIG=${RAG_CONFIG:-configs/rag.yaml}
CORPUS_CONFIG=${CORPUS_CONFIG:-configs/full.yaml}
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# qwen3:32b-q8_0 (35 GB) + KV cache for 4 x 32k contexts (~32 GB) needs ~67 GB.
gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if (( gpu_mem < ${MIN_GPU_MEM_MIB:-70000} )); then
  echo "GPU has ${gpu_mem} MiB; need >= ${MIN_GPU_MEM_MIB:-70000}. Wrong node? (submit with --nodelist=h100)"; exit 2
fi

# Free port so jobs sharing a node don't collide on 11434.
port=$(python -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1])')
export OLLAMA_HOST=127.0.0.1:$port
export OLLAMA_NUM_PARALLEL=4         # matches llm_max_async in rag.yaml
export OLLAMA_KEEP_ALIVE=-1          # keep the model loaded for the whole job
export OLLAMA_FLASH_ATTENTION=1      # smaller KV cache at num_ctx 32768
export HF_HUB_OFFLINE=1

ollama serve >"logs/ollama-$SLURM_JOB_ID.log" 2>&1 & srv=$!
trap 'kill $srv 2>/dev/null || true' EXIT
for _ in $(seq 120); do ollama list >/dev/null 2>&1 && break; sleep 1; done
ollama list

# Same config, but pointed at this job's port.
cfg=$(mktemp --suffix=.yaml)
sed "s#^\(\s*ollama_host:\).*#\1 http://$OLLAMA_HOST#" "$RAG_CONFIG" > "$cfg"
grep ollama_host "$cfg"

# Load the model onto the GPU once and confirm it is on the GPU, not CPU.
llm=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['backend']['llm_model'])" "$cfg")
ollama run "$llm" "Reply with OK." >/dev/null
ollama ps

for step in $STEPS; do
  echo "=== $step $(date)"
  case $step in
    corpus)  python -m corpus_gen.run --config "$CORPUS_CONFIG" --set "llm.host=http://$OLLAMA_HOST" ;;
    ingest)  python -m acl_rag.ingest  --rag-config "$cfg" ${INGEST_ARGS:-} ;;
    metrics) python -m acl_rag.metrics --rag-config "$cfg" ;;
    leakage) python -m acl_rag.leakage --rag-config "$cfg" ;;
    check)   python -m acl_rag.check_index --rag-config "$cfg" ;;
    C1|C2|C3|C4) python -m acl_rag.run --rag-config "$cfg" --config "$step" ;;
    *) echo "unknown step $step"; exit 1 ;;
  esac
done
echo "=== done $(date)"
