"""Train a CADRE world model on the latents of a frozen CADRE autoencoder.

    torchrun --nnodes 2 --nproc_per_node 8 main_world_model.py --env robot_manipulation \\
        --autoencoder kyutai/cadre-8x16x16 --data <root> --output_dir runs/robot_manipulation_8x16x16

--env names a recipe of configs/world_model/ (robot_manipulation, driving, indoor_navigation, human_manipulation)
or is the path of a recipe file. --autoencoder is a release autoencoder checkpoint: a Hub repo id
(kyutai/cadre-<cell>) or the final checkpoint.pt of a main_autoencoder.py run; every world-model checkpoint embeds it.
--data is a dataset of the recipe's environment in the format of DATA.md, with actions; the world model trains on its
train split.
Relaunching with the same --output_dir resumes from its latest checkpoint.
"""

import argparse
from pathlib import Path

import torch
import yaml

from dataset import EnvironmentDataset
from engine_world_model import Trainer, WorldModelTrainConfig, model_config, train
from model_autoencoder import CADRE
from model_world_model import WorldModel
from util import distributed, hub
from util.checkpoint import resume_source

RECIPES = Path(__file__).resolve().parent / "configs" / "world_model"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, help="a recipe of configs/world_model/, by name or path")
    parser.add_argument(
        "--autoencoder", required=True, help="release checkpoint of the CADRE autoencoder (.pt, or a Hub repo id)"
    )
    parser.add_argument("--data", required=True, help="dataset root (DATA.md), with actions")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--continue_from", help="a checkpoint (or run) directory whose full training state to continue")
    parser.add_argument("--batch_size", type=int, help="global batch size, overriding the recipe")
    parser.add_argument("--num_workers", type=int, help="data loader workers per GPU, overriding the recipe")
    args = parser.parse_args()

    path = Path(args.env) if args.env.endswith(".yaml") else RECIPES / f"{args.env}.yaml"
    recipe = yaml.safe_load(path.read_text())
    recipe |= {k: getattr(args, k) for k in ("batch_size", "num_workers") if getattr(args, k) is not None}
    config = WorldModelTrainConfig.model_validate(recipe)
    rank, _, device = distributed.setup()
    torch.manual_seed(config.seed + rank)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = resume_source(output_dir, args.continue_from, None)
    autoencoder_checkpoint = hub.load_checkpoint(args.autoencoder)
    model = WorldModel(model_config(config, autoencoder_checkpoint)).to(device)
    with torch.device(device):
        autoencoder = CADRE.from_checkpoint(autoencoder_checkpoint)  # in eval mode, as the world model needs it
    if distributed.is_main():
        run = {
            "env": args.env,
            "autoencoder": args.autoencoder,
            "data": args.data,
            "train": config.model_dump(mode="json"),
            "model": model.config.model_dump(mode="json"),
        }
        (output_dir / "config.yaml").write_text(yaml.safe_dump(run, sort_keys=False))

    dataset = EnvironmentDataset(config.environment, args.data, "train", config.frames, autoencoder.config.resolution)
    train(Trainer(model, autoencoder, autoencoder_checkpoint, config, device), dataset, output_dir, source)
    distributed.cleanup()


if __name__ == "__main__":
    main()
