import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Tuple, Union, Literal
from pathlib import Path
import yaml


class ActivationFunction(nn.Module):
    def __init__(self, activation: str):
        super().__init__()
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        elif activation == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        return self.act(x)


class HuBERTECGClassifier(nn.Module):
    def __init__(
        self,
        repo_id: str,
        num_labels: int,
        revision: str = "a6d3f2cbbdf7d9a3020983ab544d2ec1e265dea1",  # state used in the paper
        cache_dir: str = "./.cache",
        classifier_hidden_size: int = None,
        activation: str = "tanh",
        use_label_embedding: bool = False,
        classifier_dropout_prob: float = 0.1,
    ):
        super().__init__()

        # load pretrained HuBERT-ECG backbone from Hugging Face
        self.config = AutoConfig.from_pretrained(
            repo_id, trust_remote_code=True, revision=revision, cache_dir=cache_dir
        )
        self.hubert_ecg = AutoModel.from_pretrained(
            repo_id, trust_remote_code=True, revision=revision, cache_dir=cache_dir
        )

        # reset masking (not needed for classification)
        self.hubert_ecg.config.mask_time_prob = 0.0
        self.hubert_ecg.config.mask_feature_prob = 0.0

        self.num_labels = num_labels
        self.classifier_hidden_size = classifier_hidden_size
        self.activation = ActivationFunction(activation)
        self.use_label_embedding = use_label_embedding
        self.classifier_dropout = nn.Dropout(classifier_dropout_prob)

        # build classifier head
        if use_label_embedding:
            self.label_embedding = nn.Embedding(num_labels, self.config.hidden_size)
        else:
            if classifier_hidden_size is None:
                self.classifier = nn.Linear(self.config.hidden_size, num_labels)
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(self.config.hidden_size, classifier_hidden_size),
                    self.activation,
                    nn.Linear(classifier_hidden_size, num_labels),
                )

    def set_feature_extractor_trainable(self, trainable: bool):
        """Enable/disable conv feature extractor training"""
        self.hubert_ecg.feature_extractor.requires_grad_(trainable)

    def set_transformer_blocks_trainable(self, n_blocks: int):
        """Unfreeze only the last `n_blocks` transformer encoder layers"""
        assert 0 <= n_blocks <= self.hubert_ecg.config.num_hidden_layers
        self.hubert_ecg.encoder.requires_grad_(False)
        for i in range(1, n_blocks + 1):
            self.hubert_ecg.encoder.layers[-i].requires_grad_(True)

    def get_logits(self, pooled_output: torch.Tensor):
        """Cosine similarity with label embeddings"""
        return torch.cosine_similarity(
            pooled_output.unsqueeze(1), self.label_embedding.weight.unsqueeze(0), dim=-1
        )

    def forward(
        self,
        x: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, Union[Tuple, BaseModelOutput]]:

        return_dict = (
            return_dict
            if return_dict is not None
            else self.hubert_ecg.config.use_return_dict
        )
        output_hidden_states = (
            True if self.hubert_ecg.config.use_weighted_layer_sum else output_hidden_states
        )

        encodings = self.hubert_ecg(
            x,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_dict:
            x = encodings.last_hidden_state
        else:
            x = encodings[0] 

        # mean pooling
        if attention_mask is None:
            x = x.mean(dim=1)
        else:
            padding_mask = self.hubert_ecg._get_feature_vector_attention_mask(
                x.shape[1], attention_mask
            )
            x[~padding_mask] = 0.0
            x = x.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        x = self.classifier_dropout(x)

        return {
            "out": self.get_logits(x) if self.use_label_embedding else self.classifier(x),
            "encodings": encodings
        }


def build_classifier(
    model_store: str,
    status: str = 'pretrained',
    device: str = 'cpu',
    checkpoint: Optional[str] = None,
    classifier_hidden_size: int = None,
    num_labels: int = 26,
    activation: str = "tanh",
    use_label_embedding: bool = False,
    classifier_dropout_prob: float = 0.1,
    size: Literal["small", "base", "large"] = "small"
):
    """
    Build a HuBERT ECG classifier.

    Args:
        model_store (str): Local directory to store model files.
        status (str): "pretrained" to download Hugging Face model, "custom" to load from checkpoint.
        checkpoint (str): Path to a custom checkpoint (required if status != "pretrained").

    Returns:
        model (torch.nn.Module): The classifier model.
    """

    model_store = Path(model_store)
    cache_dir = model_store / f"HuBERTECG" / "pretrained"
    repo_id = f"Edoardo-BS/hubert-ecg-{size}"

    model = HuBERTECGClassifier(
        repo_id=repo_id,
        num_labels=num_labels,
        cache_dir=cache_dir,
        classifier_hidden_size=classifier_hidden_size,
        activation=activation,
        use_label_embedding=use_label_embedding,
        classifier_dropout_prob=classifier_dropout_prob,
    )

    if status == "finetuned":
        if checkpoint is None:
            raise ValueError("You must provide a checkpoint path if status != 'pretrained'.")
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)

    elif status != "pretrained":
        raise ValueError("status must be 'pretrained' or 'finetuned'")

    return model
