# Hy3 three-way parity

The parity suite compares the same deterministic two-layer BF16 proxy across:

- Hugging Face `HYV3ForCausalLM`;
- the native Megatron Lite Hy3 implementation;
- NVIDIA Megatron Bridge `HYV3Bridge`.

It saves the Hugging Face proxy as a safetensors checkpoint, loads that
checkpoint through each implementation's production weight-conversion path,
exports both native models back to Hugging Face names, checks complete key
coverage, and runs a fixed-token forward pass.

## Frozen references

| Component | Revision |
| --- | --- |
| `tencent/Hy3` | `a960ebc3da325ba167f069f76c41eb62c9280d22` |
| `NVIDIA-NeMo/Megatron-Bridge` | `fc704f7809f84461c6e2cbf476acb28c4a5654c0` |
| Megatron Core | `6cd6ea530e18776da54297bbf88292264264bcd3` |
| Transformers | `5.12.1` |

## Results

All 39 checkpoint tensors had identical names, shapes, dtypes, and values in
all three pairwise comparisons.

| Comparison | Weight max abs | Weight max rel | Logits max abs | Logits max rel |
| --- | ---: | ---: | ---: | ---: |
| Megatron Lite vs Hugging Face | 0 | 0 | 0.00048828125 | 0.0046728970 |
| Megatron Bridge vs Hugging Face | 0 | 0 | 0.00048828125 | 0.0046728970 |
| Megatron Lite vs Megatron Bridge | 0 | 0 | 0.00006103515625 | 0.0005841121 |

`max_rel` is the L-infinity absolute error divided by the L-infinity
reference magnitude. The BF16 logits checks use the existing Hy3 proxy
tolerance of `atol=rtol=0.03`.

Run the configuration contract without a GPU:

```bash
python -m pytest -q -s \
  tests/smoke/test_hy3_three_way_parity.py \
  -k megatron_bridge_hy3_config_contract
```

Run the weight and logits comparison in a single-GPU distributed environment:

```bash
python -m pytest -q -s \
  tests/smoke/test_hy3_three_way_parity.py \
  -k weight_and_logits_parity
```
