"""Evaluate a world model's rollouts or an autoencoder's reconstructions.

    torchrun --nproc_per_node=8 main_eval.py rollout --env robot_manipulation \\
        --checkpoint kyutai/cadre-world-models-2x16x16/robot_manipulation \\
        --probe kyutai/cadre-action-probes/robot_manipulation \\
        --data <root> --manifest manifests/robot_manipulation_test.txt
    torchrun --nproc_per_node=8 main_eval.py reconstruction --checkpoint kyutai/cadre-2x16x16 \\
        --data <OpenVid-HD root> --manifest manifests/openvid_hd_test.txt

--checkpoint and --probe take a Hub repo id, optionally followed by a subfolder (as above), or a release .pt file.

--data is a dataset in the format of DATA.md whose `test` split holds the manifest's videos (manifests/README.md);
for a rollout, a dataset of the environment --env (dataset.ENVIRONMENTS), which also sets the reported numbers: ARR
on the 12 wrist dimensions for human_manipulation, PSNR at 48 frames for indoor_navigation. Any number of GPUs gives
the same rollouts (engine_eval). Rank 0 prints the reported numbers and writes every metric, with the protocol, to
--output as JSON; `--set decode_steps=2` overrides a setting of the --config recipe.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml

from dataset import ENVIRONMENTS, ClipDataset, EnvironmentDataset, read_manifest
from engine_eval import (
    ReconstructionEvalConfig,
    RolloutEvalConfig,
    evaluate_reconstruction,
    evaluate_rollout,
    paper_numbers,
    shard_loader,
)
from metrics import ARR_DIMS, SLICE_FRAMES, ReconstructionMetrics, RolloutMetrics
from model_autoencoder import CADRE
from model_idm import load_probe
from model_world_model import load_checkpoint
from util import distributed

CONFIGS = {
    "rollout": (RolloutEvalConfig, "configs/eval/rollout.yaml"),
    "reconstruction": (ReconstructionEvalConfig, "configs/eval/reconstruction.yaml"),
}


def parse_set(pairs: list[str]) -> dict:
    """["key=value", ...] -> {key: value parsed as YAML}."""
    return {k: yaml.safe_load(v) for k, v in (p.split("=", 1) for p in pairs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=sorted(CONFIGS))
    parser.add_argument("--checkpoint", required=True, help="world model (rollout) or autoencoder: repo id or .pt")
    parser.add_argument("--probe", help="the environment's IDM probe (rollout), repo id or .pt: adds ARR")
    parser.add_argument("--data", required=True, help="dataset root (DATA.md)")
    parser.add_argument("--manifest", required=True, help='"<key> <start frame>" windows of the split')
    parser.add_argument("--split", default="test", help="the dataset split the manifest's videos are in")
    parser.add_argument("--env", choices=list(ENVIRONMENTS), help="the environment of the rollout's world model")
    parser.add_argument("--config", help="recipe (default: configs/eval/<task>.yaml)")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override recipe settings")
    parser.add_argument("--output", help="JSON file of the results (default: eval_<checkpoint>_<manifest>.json)")
    args = parser.parse_args()

    config_class, default = CONFIGS[args.task]
    recipe = yaml.safe_load(Path(args.config or Path(__file__).parent / default).read_text())
    config = config_class.model_validate(recipe | parse_set(args.set))
    _, world_size, device = distributed.setup()

    output = Path(args.output or f"eval_{Path(args.checkpoint).stem}_{Path(args.manifest).stem}.json")
    windows = read_manifest(args.manifest)
    if args.task == "rollout":
        assert args.env, "a rollout needs its --env"
        world_model, autoencoder_checkpoint = load_checkpoint(args.checkpoint, device)
        world_model.eval()
        assert world_model.config.action_dim == ENVIRONMENTS[args.env].action_dim, f"not a world model of {args.env}"
        with torch.device(device):
            autoencoder = CADRE.from_checkpoint(autoencoder_checkpoint)
        probe = load_probe(args.probe).to(device) if args.probe else None
        if config.compile:
            world_model.velocity = torch.compile(world_model.velocity)
            autoencoder.decoder.flow_forward = torch.compile(autoencoder.decoder.flow_forward)
        resolution = autoencoder.config.resolution
        dataset = EnvironmentDataset(args.env, args.data, args.split, config.clip_frames, resolution, args.manifest)
        metrics = RolloutMetrics(config.generated_frames // SLICE_FRAMES, probe, ARR_DIMS.get(args.env), device)
        loader = shard_loader(dataset, args.manifest, config.seed, config.batch_size, config.num_workers)
        results = evaluate_rollout(world_model, autoencoder, metrics, loader, config, device)
        table = paper_numbers(results, args.env)
    else:
        with torch.device(device):
            autoencoder = CADRE.from_checkpoint(args.checkpoint)
        dataset = ClipDataset(
            args.data, args.split, config.clip_frames, autoencoder.config.resolution, args.manifest, actions=False
        )
        metrics = ReconstructionMetrics(device)
        loader = shard_loader(dataset, args.manifest, config.seed, config.batch_size, config.num_workers)
        results = table = evaluate_reconstruction(autoencoder, metrics, loader, config, device)

    if distributed.is_main():
        out = {
            "task": args.task,
            "checkpoint": str(args.checkpoint),
            "probe": args.probe,
            "manifest": str(args.manifest),
            "env": args.env,
            "windows": len(windows),
            "distinct_windows": len(set(windows)),
            "world_size": world_size,
            "config": config.model_dump(),
            "paper": table,
            "metrics": results,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(out, indent=2))
        width = max(map(len, table))
        print("\n".join(f"{k:<{width}}  {v:.4f}" for k, v in table.items()))
        print(f"{len(windows)} windows ({len(set(windows))} distinct) on {world_size} GPU(s) -> {output}")
    distributed.cleanup()


if __name__ == "__main__":
    main()
