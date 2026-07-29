#!/usr/bin/env bash
# One-GPU acceptance for Hy3 routed-expert MXFP4 QAT.
# Submit with the site-specific account/partition flags required by your cluster.
#SBATCH --job-name=hy3-mxfp4-qat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

set -euo pipefail

: "${HY3_REPO:?set to the Hy3 checkout under test}"
: "${MLITE_REPO:?set to the Megatron-LM checkout under test}"
: "${HY3_REAL_CHECKPOINT:?set to the real Tencent Hy3 HF checkpoint}"
: "${MODELOPT_SITE:?set to the ModelOpt Python site-packages directory}"
: "${CONTAINER_IMAGE:?set to the validated MLite container image}"
: "${EXPECTED_HY3_COMMIT:?set to the exact Hy3 commit under test}"
: "${EXPECTED_MLITE_COMMIT:?set to the exact Megatron-LM commit under test}"

actual_hy3_commit="$(git -C "$HY3_REPO" rev-parse HEAD)"
actual_mlite_commit="$(git -C "$MLITE_REPO" rev-parse HEAD)"
test "$actual_hy3_commit" = "$EXPECTED_HY3_COMMIT" || {
  echo "HY3_COMMIT_MISMATCH expected=$EXPECTED_HY3_COMMIT actual=$actual_hy3_commit" >&2
  exit 64
}
test "$actual_mlite_commit" = "$EXPECTED_MLITE_COMMIT" || {
  echo "MLITE_COMMIT_MISMATCH expected=$EXPECTED_MLITE_COMMIT actual=$actual_mlite_commit" >&2
  exit 65
}
for path in "$HY3_REPO" "$MLITE_REPO" "$HY3_REAL_CHECKPOINT" "$MODELOPT_SITE" "$CONTAINER_IMAGE"; do
  test -e "$path" || {
    echo "HY3_QAT_REQUIRED_PATH_MISSING $path" >&2
    exit 66
  }
done

echo "HY3_QAT_MODULES hy3=$actual_hy3_commit mlite=$actual_mlite_commit"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$HY3_REPO/src:$MLITE_REPO/experimental/lite:$MODELOPT_SITE${PYTHONPATH:+:$PYTHONPATH}"
export HY3_REPO MLITE_REPO HY3_REAL_CHECKPOINT

srun \
  --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre \
  --export=ALL \
  bash -lc '
    set -euo pipefail
    nvidia-smi --query-gpu=name,uuid,compute_cap --format=csv,noheader
    python - <<'"'"'PY'"'"'
from pathlib import Path
import os
import mlite_hy3
import megatron.lite

hy3_source = Path(mlite_hy3.__file__).resolve()
mlite_source = Path(megatron.lite.__file__).resolve()
assert hy3_source.is_relative_to(Path(os.environ["HY3_REPO"]).resolve())
assert mlite_source.is_relative_to(
    Path(os.environ["MLITE_REPO"]).resolve() / "experimental/lite"
)
print(f"HY3_QAT_IMPORTS hy3={hy3_source} mlite={mlite_source}", flush=True)
PY
    python -m pytest -q -s \
      "$HY3_REPO/tests/smoke/test_hy3_mxfp4_qat.py"
  '
