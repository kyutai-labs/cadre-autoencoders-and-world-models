"""Train the inverse dynamics probe of one environment.

    torchrun --nproc_per_node=8 main_idm.py --env robot_manipulation --data <dataset> \\
        --dino_weights <dinov3_vitb16 .pth> --output_dir runs/idm_robot_manipulation

--data is a dataset of the environment in the format of DATA.md, the one its world models train on; the probe reads
its continuous actions. --batch_size is per GPU: with fewer GPUs, raise --grad_accum to keep the clips per step.
The released probes are the step-55000 checkpoints (human_manipulation: the final one, step 60000).
"""

import argparse
import os

import torch
import torch.distributed as dist

from dataset import ENVIRONMENTS, EnvironmentDataset
from engine_idm import IDMTrainConfig, train
from model_autoencoder import load_dinov3
from model_idm import IDMConfig, InverseDynamicsProbe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", required=True, choices=list(ENVIRONMENTS))
    parser.add_argument("--data", required=True, help="the environment's dataset (DATA.md; train and val splits)")
    parser.add_argument("--dino_weights", required=True, help="Meta's dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", help="a probe_step<k>.pt of this run")
    parser.add_argument("--total_steps", type=int)
    parser.add_argument("--batch_size", type=int, help="per GPU")
    parser.add_argument("--grad_accum", type=int)
    parser.add_argument("--num_workers", type=int)
    args = parser.parse_args()

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    overrides = {k: getattr(args, k) for k in ("total_steps", "batch_size", "grad_accum", "num_workers")}
    config = IDMTrainConfig(**{k: v for k, v in overrides.items() if v is not None})

    window = IDMConfig.model_fields["window"].default
    train_set = EnvironmentDataset(args.env, args.data, "train", window)
    val_set = EnvironmentDataset(args.env, args.data, "val", window)
    torch.manual_seed(config.seed)
    dino = load_dinov3()
    dino.load_state_dict(torch.load(args.dino_weights, map_location="cpu", weights_only=True))
    probe = InverseDynamicsProbe(IDMConfig(action_dim=ENVIRONMENTS[args.env].action_dim), dino).to(device)
    train(probe, train_set, val_set, config, args.output_dir, resume=args.resume)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
