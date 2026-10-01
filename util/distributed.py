"""Process-group setup and the few collectives the trainers use. One process per GPU under torchrun, or a single
process without a process group."""

import datetime
import os

import torch
import torch.distributed as dist


def setup() -> tuple[int, int, torch.device]:
    """(rank, world_size, device). Under torchrun, joins the NCCL group and pins this process to its GPU."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # torch.compile's first step takes minutes and skews across ranks, longer than the default NCCL timeouts.
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank), timeout=datetime.timedelta(hours=1))
    return dist.get_rank(), dist.get_world_size(), torch.device("cuda", local_rank)


def rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def broadcast_object(obj, src: int = 0):
    """Rank src's obj on every rank."""
    if not dist.is_initialized():
        return obj
    box = [obj]
    dist.broadcast_object_list(box, src=src)
    return box[0]


def all_reduce_mean_(tensor: torch.Tensor) -> torch.Tensor:
    """Average `tensor` over the ranks, in place."""
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


def all_reduce_sum_(tensor: torch.Tensor) -> torch.Tensor:
    if dist.is_initialized():
        dist.all_reduce(tensor)
    return tensor


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
