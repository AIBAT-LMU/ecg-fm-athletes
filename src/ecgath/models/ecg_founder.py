import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
from ecgath.models.third_party.ecg_founder import Net1D
from huggingface_hub import hf_hub_download


class ECGFounderClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        num_labels: int = 26,
        base_filters: int = 64,
        ratio: int = 1,
        filter_list: list = [64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list: list = [2, 2, 2, 3, 3, 4, 4],
        kernel_size: int = 16,
        stride: int = 2,
        groups_width: int = 16,
        use_bn: bool = False,
        use_do: bool = False,
        verbose: bool = False,
    ):
        super().__init__()

        self.model = Net1D(
            in_channels=in_channels,
            base_filters=base_filters,
            ratio=ratio,
            filter_list=filter_list,
            m_blocks_list=m_blocks_list,
            kernel_size=kernel_size,
            stride=stride,
            groups_width=groups_width,
            n_classes=num_labels,
            use_bn=use_bn,
            use_do=use_do,
            verbose=verbose,
        )

        # Add classification head
        self.model.dense = nn.Linear(self.model.dense.in_features, num_labels)


    def set_backbone_trainable(self, trainable: bool = True):
        """
        Set trainable layers of the model.

        Args:
            backbone_trainable (bool): If True, allows training of the backbone.
                                        If False, freezes backbone weights.
        """
        # Freeze or unfreeze backbone
        for name, param in self.model.named_parameters():
            if not name.startswith("dense"):
                param.requires_grad = backbone_trainable

        # Always keep classification head trainable
        for param in self.model.dense.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor):
        return {'out': self.model(x)}


def build_classifier(
    model_store: str,
    status: str = "pretrained",
    checkpoint: Optional[str] = None,
    device: str = "cpu",
    in_channels: int = 12,
    num_labels: int = 26,
    base_filters: int = 64,
    ratio: int = 1,
    filter_list: list = [64, 160, 160, 400, 400, 1024, 1024],
    m_blocks_list: list = [2, 2, 2, 3, 3, 4, 4],
    kernel_size: int = 16,
    stride: int = 2,
    groups_width: int = 16,
    use_bn: bool = False,
    use_do: bool = False,
    verbose: bool = False,
):
    """
    Build an ECGFounder classifier.

    Args:
        status (str): "pretrained" to download from Hugging Face, "finetuned" to load a custom checkpoint.
        checkpoint (str): Path to a custom checkpoint (required if status="finetuned").
        device (str): Device to map model/weights.

    Returns:
        model (torch.nn.Module)
    """

    model_store = Path(model_store)
    cache_dir = model_store / "ECGFounder" / "pretrained"
    repo_id = "PKUDigitalHealth/ECGFounder"

    model = ECGFounderClassifier(
        in_channels=in_channels,
        num_labels=num_labels,
        base_filters=base_filters,
        ratio=ratio,
        filter_list=filter_list,
        m_blocks_list=m_blocks_list,
        kernel_size=kernel_size,
        stride=stride,
        groups_width=groups_width,
        use_bn=use_bn,
        use_do=use_do,
        verbose=verbose,
    )

    if status == "pretrained":
        if in_channels not in [1, 12]:
            raise ValueError("in_channels must be 1 or 12")

        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{in_channels}_lead_ECGFounder.pth",
            repo_type="model",
            revision="d9b1793951b2342f5f7e84f1ac03cd37f8a08724",  # state used in the paper
            cache_dir=cache_dir
        )

        # Load checkpoint safely
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint.get('state_dict', checkpoint)
        # Exclude dense layer weights to allow new classification head
        state = {k: v for k, v in state.items() if not k.startswith('dense.')}
        model.load_state_dict(state, strict=False)

    elif status == "finetuned":
        if checkpoint is None:
            raise ValueError("You must provide a checkpoint path if status='finetuned'.")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=device)
        state = state.get('state_dict', state)
        model.load_state_dict(state)

    else:
        raise ValueError("status must be 'pretrained' or 'finetuned'")

    return model.to(device)