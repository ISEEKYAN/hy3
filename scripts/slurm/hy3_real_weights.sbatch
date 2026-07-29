#!/usr/bin/env bash
# CPU-only preparation of the minimal real Hy3 weight subset used by QAT parity.
# Submit with the site-specific account/partition flags required by your cluster.
#SBATCH --job-name=hy3-real-weights
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail

: "${HY3_REPO:?set to the Hy3 checkout under test}"
: "${MLITE_REPO:?set to the Megatron-LM checkout under test}"
: "${HY3_REAL_CHECKPOINT:?set to the output checkpoint subset directory}"
: "${PYTHON_DEPS:?set to a Python site with huggingface_hub and safetensors}"
: "${CONTAINER_IMAGE:?set to the validated MLite container image}"
: "${EXPECTED_HY3_COMMIT:?set to the exact Hy3 commit under test}"
: "${EXPECTED_MLITE_COMMIT:?set to the exact Megatron-LM commit under test}"

actual_hy3_commit="$(git -C "$HY3_REPO" rev-parse HEAD)"
actual_mlite_commit="$(git -C "$MLITE_REPO" rev-parse HEAD)"
test "$actual_hy3_commit" = "$EXPECTED_HY3_COMMIT" || exit 64
test "$actual_mlite_commit" = "$EXPECTED_MLITE_COMMIT" || exit 65
echo "HY3_REAL_WEIGHT_MODULES hy3=$actual_hy3_commit mlite=$actual_mlite_commit"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$HY3_REPO/src:$MLITE_REPO/experimental/lite:$PYTHON_DEPS"
srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --export=ALL \
  python "$HY3_REPO/scripts/prepare_hy3_real_weights.py" \
    --revision a960ebc3da325ba167f069f76c41eb62c9280d22 \
    --output "$HY3_REAL_CHECKPOINT"
