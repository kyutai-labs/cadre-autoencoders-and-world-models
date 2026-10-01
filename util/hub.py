"""Hugging Face layout of a release checkpoint: a folder holding model.safetensors (its state_dict) and config.json (its
other fields), in a local directory or a Hub repo, optionally in a subfolder of either. load_checkpoint reads any
release checkpoint: a .pt file, such a folder, or a repo id such as "kyutai/cadre-world-models-2x16x16/driving".
"""

import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

CONFIG, WEIGHTS = "config.json", "model.safetensors"
SCALARS = ("latent_mean", "latent_std")  # float64 0-d tensors in the .pt checkpoints, numbers in config.json


def save_pretrained(checkpoint: dict, directory: str | Path) -> None:
    """A release checkpoint dict -> directory/{model.safetensors, config.json}; 0-d tensors (the latent statistics)
    are written to the JSON as numbers."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file(checkpoint["state_dict"], directory / WEIGHTS, metadata={"format": "pt"})
    config = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in checkpoint.items() if k != "state_dict"}
    (directory / CONFIG).write_text(json.dumps(config, indent=2) + "\n")


def load_pretrained(path_or_repo_id: str | Path, subfolder: str | None = None, device="cpu") -> dict:
    """The checkpoint dict of save_pretrained, read from a local directory or downloaded from a Hub repo id (only
    `subfolder`, when given), with the state_dict on `device`."""
    root = Path(path_or_repo_id)
    if not root.is_dir():
        root = Path(snapshot_download(str(path_or_repo_id), allow_patterns=[f"{subfolder}/*"] if subfolder else None))
    directory = root / subfolder if subfolder else root
    checkpoint = json.loads((directory / CONFIG).read_text())
    checkpoint |= {k: torch.tensor(checkpoint[k], dtype=torch.float64) for k in SCALARS if k in checkpoint}
    return checkpoint | {"state_dict": load_file(directory / WEIGHTS, device=str(device))}


def load_checkpoint(source: str | Path, device="cpu") -> dict:
    """The release checkpoint dict of `source`, as its .pt file holds it: a .pt file, a Hugging Face folder, or a Hub
    repo id optionally followed by a subfolder ("kyutai/cadre-2x16x16", "kyutai/cadre-world-models-2x16x16/driving").
    A world model's autoencoder, a sibling folder in the Hugging Face layout, is nested back in."""
    path = Path(source)
    if path.is_file() or path.suffix in (".pt", ".pth"):
        return torch.load(path, map_location=device, weights_only=True)
    if path.is_dir():
        root, subfolder = path.parent, path.name
    else:
        owner, name, *subfolder = str(source).split("/")
        root, subfolder = f"{owner}/{name}", "/".join(subfolder) or None
    checkpoint = load_pretrained(root, subfolder, device)
    if isinstance(checkpoint.get("autoencoder"), str):
        checkpoint["autoencoder"] = load_pretrained(root, checkpoint["autoencoder"], device)
    return checkpoint
