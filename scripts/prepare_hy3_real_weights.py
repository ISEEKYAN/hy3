"""Materialize the smallest real Hy3 checkpoint subset for MXFP4 parity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors import safe_open

from mlite_hy3.config import Hy3Config
from mlite_hy3.lite.checkpoint import Hy3WeightSpec


def _link(source: str, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(Path(source).resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--repo", default="tencent/Hy3")
    parser.add_argument("--minimum-elements", type=int, default=99_090_432)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    common = {"repo_id": args.repo, "revision": args.revision}
    config_path = hf_hub_download(filename="config.json", **common)
    index_path = hf_hub_download(
        filename="model.safetensors.index.json",
        **common,
    )
    _link(config_path, args.output / "config.json")
    _link(index_path, args.output / "model.safetensors.index.json")

    config = Hy3Config.from_hf(str(args.output))
    spec = Hy3WeightSpec(config)
    expected = sorted(
        hf_name
        for native_name, hf_names in spec.weight_map().items()
        if spec.is_expert(native_name)
        for hf_name in hf_names
    )
    with open(index_path) as handle:
        index = json.load(handle)["weight_map"]
    missing = sorted(set(expected) - index.keys())
    if missing:
        raise RuntimeError(
            f"official checkpoint is missing {len(missing)} routed weights: "
            f"{missing[:3]}"
        )

    shard_to_names: dict[str, list[str]] = {}
    for name in expected:
        shard_to_names.setdefault(index[name], []).append(name)

    elements = 0
    tensors = 0
    shards = 0
    for shard, names in shard_to_names.items():
        shard_path = hf_hub_download(filename=shard, **common)
        _link(shard_path, args.output / shard)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                elements += tensor.numel()
                tensors += 1
        shards += 1
        print(
            f"HY3_REAL_WEIGHT_PREP shard={shard} shards={shards} "
            f"tensors={tensors} elements={elements}",
            flush=True,
        )
        if elements >= args.minimum_elements:
            break

    if elements < args.minimum_elements:
        raise RuntimeError(
            f"only found {elements} routed-expert elements, "
            f"need {args.minimum_elements}"
        )
    manifest = {
        "elements": elements,
        "repo": args.repo,
        "revision": args.revision,
        "shards": shards,
        "tensors": tensors,
    }
    with open(args.output / "hy3-real-weight-manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write(os.linesep)
    print("HY3_REAL_WEIGHT_READY=" + json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
