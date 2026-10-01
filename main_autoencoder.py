"""Train a CADRE autoencoder.

    torchrun --nnodes 2 --nproc_per_node 8 main_autoencoder.py --model cadre_2x32x32 \\
        --config configs/autoencoder/pretrain.yaml --data <root> [<root> ...] [--data_weights <w> ...] \\
        --dino_weights dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth --output_dir runs/cadre_2x32x32

--data takes one or more datasets in the format of DATA.md (their train splits), mixed by --data_weights (default:
equal). The recipes, in order:
    configs/autoencoder/pretrain.yaml        150k steps
    configs/autoencoder/cooldown_360k.yaml   --continue_from runs/<cell>/checkpoint-149999, to 360k
    configs/autoencoder/multires_ft.yaml     --continue_from runs/<cell>_360k/checkpoint-359999, 12 h
Relaunching with the same --output_dir resumes from its latest checkpoint.
"""

import argparse
from pathlib import Path

import torch
import yaml

from dataset import ClipDataset
from engine_autoencoder import AutoencoderTrainConfig, Trainer, data_resolution, model_config, train
from engine_eval import reconstruction_evaluator
from model_autoencoder import CADRE, CADRE_MODELS
from util import distributed
from util.checkpoint import resume_source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(CADRE_MODELS))
    parser.add_argument("--config", required=True, help="a recipe of configs/autoencoder/")
    parser.add_argument("--data", required=True, nargs="+", help="dataset roots (DATA.md)")
    parser.add_argument("--data_weights", type=float, nargs="+", help="mixture weights, one per root")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dino_weights", help="Meta's DINOv3 ViT-B/16 file; needed by a run that starts fresh")
    parser.add_argument("--continue_from", help="a checkpoint (or run) directory whose full training state to continue")
    parser.add_argument(
        "--finetune_from", help="a release checkpoint (.pt, or a Hub repo id) whose weights to start from, at step 0"
    )
    parser.add_argument("--batch_size", type=int, help="global batch size, overriding the recipe")
    parser.add_argument("--num_workers", type=int, help="data loader workers per GPU, overriding the recipe")
    parser.add_argument("--eval_data", help="dataset root (DATA.md) whose val clips are scored every eval_every steps")
    parser.add_argument("--eval_clips", type=int, default=256, help="number of those val clips")
    args = parser.parse_args()
    weights = args.data_weights or [1.0] * len(args.data)
    assert len(weights) == len(args.data), "give one --data_weights value per --data root"

    recipe = yaml.safe_load(Path(args.config).read_text())
    recipe |= {k: getattr(args, k) for k in ("batch_size", "num_workers") if getattr(args, k) is not None}
    config = AutoencoderTrainConfig.model_validate(recipe)
    rank, _, device = distributed.setup()
    torch.manual_seed(config.seed + rank)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = resume_source(output_dir, args.continue_from, args.finetune_from)
    model = CADRE(model_config(args.model, config))
    if source is None:
        assert args.dino_weights, "a run that starts fresh needs --dino_weights"
        model.encoder.dino.load_state_dict(torch.load(args.dino_weights, map_location="cpu", weights_only=True))
    model.to(device)
    if distributed.is_main():
        run = {"model": args.model, "data": args.data, "data_weights": weights, "train": config.model_dump(mode="json")}
        (output_dir / "config.yaml").write_text(yaml.safe_dump(run, sort_keys=False))

    resolution = data_resolution(model, config)
    datasets = [ClipDataset(root, "train", model.config.num_frames, resolution, actions=False) for root in args.data]
    evaluate = reconstruction_evaluator(args.eval_data, args.eval_clips, output_dir, device) if args.eval_data else None
    train(Trainer(model, args.model, config, device), datasets, weights, output_dir, source, evaluate=evaluate)
    distributed.cleanup()


if __name__ == "__main__":
    main()
