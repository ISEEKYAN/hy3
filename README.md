# hy3

## What this repository is

`mlite-hy3` packages the public [`tencent/Hy3`](https://huggingface.co/tencent/Hy3) architecture as an **external Megatron Lite model**. It is both a usable model package and a worked example of adding a model without editing Megatron Lite's built-in registry.

The package contains:

- the Hy3 architecture config and invariants;
- a native model assembled from Megatron Lite attention, expert, dispatcher, parallel, and loss primitives;
- Hy3-specific router, shared-MLP, and MTP compositions kept in this package;
- the HF/native `Hy3WeightSpec` and checkpoint entry points;
- CPU unit tests, independent Transformers parity, real checkpoint IO, short-train, and distributed smoke tests.

Importing `mlite_hy3` has no registration side effect. Applications explicitly call `register_model()` before asking Megatron Lite to resolve or build Hy3. No source file under `megatron.lite` is replaced or monkey-patched.

For expert-parallel HF export, use a Megatron Lite revision that includes the generic TE `GroupedLinear` local-to-global expert-name export fix. This fix belongs in Megatron Lite because it is model-independent; it is deliberately not copied into this repository.

The frozen reference architecture and primitive mapping are documented in [docs/architecture.md](docs/architecture.md).

## Quick start

Megatron Lite currently lives in `experimental/lite` of Megatron-LM. Put that source directory on `PYTHONPATH`, then install this package:

```bash
git clone https://github.com/ISEEKYAN/Megatron-LM.git
git clone https://github.com/ISEEKYAN/hy3.git

export PYTHONPATH="$PWD/Megatron-LM/experimental/lite:${PYTHONPATH}"
python -m pip install -e ./hy3
```

Register Hy3 before constructing a runtime:

```python
from mlite_hy3 import register_model

register_model()

from megatron.lite.model.registry import (
    get_model_package,
    get_train_runtime_module,
    resolve_model_type_from_hf,
)

assert resolve_model_type_from_hf({"model_type": "hy_v3"}) == "hy3"
assert get_model_package("hy3").Hy3Config.__name__ == "Hy3Config"
protocol = get_train_runtime_module("hy3")
```

Run the CPU checks:

```bash
cd hy3
python -m pip install -e '.[test]'
python -m pytest -q tests/unit
```

The GPU suites require CUDA, Transformer Engine, a Transformers release with
`HYV3ForCausalLM`, and an initialized distributed environment. The three-way
suite additionally requires a current
[`NVIDIA-NeMo/Megatron-Bridge`](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
checkout that contains `HYV3Bridge`. Run them through the cluster scheduler,
not on a login node:

```bash
python -m pytest -q tests/smoke/test_hy3_hf_parity.py
python -m pytest -q -s \
  tests/smoke/test_hy3_three_way_parity.py \
  -k megatron_bridge_hy3_config_contract
python -m pytest -q -s \
  tests/smoke/test_hy3_three_way_parity.py \
  -k weight_and_logits_parity

torchrun --nproc-per-node=2 -m pytest -q \
  tests/smoke/test_hy3_acceptance.py \
  -k distributed_forward_backward_uses_requested_paths
```

Set `HY3_TOPOLOGY` to `ep2`, `cp2_thd`, or `cp2_ep2`; the last option requires four ranks. The test asserts the requested topology and communication path so a fallback cannot silently pass.

See [the three-way parity report](docs/three-way-parity.md) for the frozen
references, proxy contract, and exact pairwise metrics.

## The four-stage model-support workflow

This port follows the same staged workflow used for internal and external Megatron Lite model integrations.

### 1. Freeze the independent reference

Pin the public model revision, configuration digest, weight-index digest, and
independent implementation before writing native code. For Hy3, the current
reference is Hugging Face revision
`a960ebc3da325ba167f069f76c41eb62c9280d22`; relative to the original freeze it
only adds the explicit `dtype=bfloat16` config field and leaves the weight index
unchanged. The manifest records 80 decoder layers plus one MTP layer, GQA, a
dense first layer, routed MoE layers, persistent expert-selection bias, and one
shared expert.

### 2. Map capabilities to existing primitives

Start from the Qwen3-MoE protocol shape, but select primitives by capability rather than copying a model directory. Hy3 directly reuses Megatron Lite GQA, Q/K norm, RoPE, TP/CP/THD, expert compute, token dispatch, pipeline layout, losses, recompute, and checkpoint machinery. The package owns only Hy3-specific composition and parameter semantics.

### 3. Implement config, model protocol, and weights

Keep architecture fields in `Hy3Config`, runtime choices in `ImplConfig`, and checkpoint names/layout transforms in `Hy3WeightSpec`. The explicit registration entry point maps HF `model_type=hy_v3` to the external package and protocol. Weight tests check dense, sparse, shared-expert, router-bias, MTP, QKV packing, and SwiGLU round trips.

### 4. Validate in increasing-cost stages

Validation progresses from CPU contracts to an independent HF reference, real HF save/load/export, short training, and distributed combinations with anti-fallback assertions. The following Slurm evidence was produced for the source implementation immediately before it was extracted into this standalone package:

| Job | Result | Non-skipped coverage |
| --- | --- | --- |
| `13498433` | `COMPLETED`, exit `0:0` | Transformers 5.6 independent parity: both layer outputs, router indices/scores, logits, loss, and embedding/QKV/router/head gradients; 1 test passed |
| `13498434` | `COMPLETED`, exit `0:0` | checkpoint unit tests, real safetensors save/load/re-export, and a 12-step loss-decreasing short train; 4 + 2 tests passed |
| `13498435` | `COMPLETED`, exit `0:0` | EP2 forward/backward with explicit all-to-all path assertion; target test passed on every rank |
| `13498437` | `COMPLETED`, exit `0:0` | CP2 + variable-length THD forward/backward with topology assertions; target test passed on every rank |
| `13498438` | `COMPLETED`, exit `0:0` | combined CP2 x EP2 forward/backward with THD and all-to-all assertions; target test passed on every rank |

The environment preparation job `13498357` was also `COMPLETED` with exit `0:0` and recorded NVRx 0.6.0, async support enabled, Transformers 5.6.0, and pytest 9.0.2.

These job IDs are historical validation evidence for the pre-extraction implementation, not claims that the standalone packaging commits were rerun on GPU. The extracted package must be rerun against the exact Megatron Lite revision and container selected for production. On the extraction host, the fresh local result was `13 passed, 5 skipped`; four skips required CUDA and one required safetensors.

Acceptance checklist:

- [x] Explicit external registration and HF model-type resolution
- [x] Architecture invariants and unsupported-drift rejection
- [x] Synthetic HF/native/HF weight-layout round trips
- [x] Independent Transformers layer/router/logit/loss/gradient parity
- [x] Real HF checkpoint save/load/re-export and short training
- [x] EP2, CP2+THD, and CP2xEP2 non-fallback distributed paths
- [ ] Rerun the GPU matrix for the exact downstream Megatron Lite revision and container

## Use this repository as a template for another model

1. **Freeze a reference.** Record the HF revision, config and index digests, parameter names, architecture invariants, and the independent reference implementation.
2. **Inventory Megatron Lite primitives.** Reuse attention, MoE, parallel, checkpoint, optimizer, and loss modules by capability. Put genuinely model-specific behavior in your external package instead of copying common logic into `forward()`.
3. **Create a small package.** A practical layout is:

   ```text
   src/your_model/
     __init__.py
     register.py
     config.py
     primitives.py
     lite/
       model.py
       protocol.py
       checkpoint.py
   tests/
     unit/
     smoke/
   ```

4. **Expose explicit registration.** Call Megatron Lite's public `register_model()` with your external package and protocol paths. Do not use `sitecustomize`, `.pth` injection, or runtime monkey-patching.
5. **Make `WeightSpec` exhaustive.** Check that every reference tensor is consumed exactly once or is explicitly allowlisted, and test packing, expert global/local IDs, PP layer remapping, tied weights, and export round trips.
6. **Validate in stages.** Begin with CPU/static contracts, then compare against an independent reference, then exercise real IO and backward, and finally run TP/EP/CP/PP combinations through Slurm with anti-fallback assertions.
7. **Keep framework hooks separate.** If the public registry or primitive API truly cannot support an external package, propose the smallest model-agnostic Megatron Lite change as a separate PR. Do not hide a framework patch inside the model example.
