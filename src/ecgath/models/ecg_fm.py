from pathlib import Path
from fairseq_signals.models import build_model_from_checkpoint
from huggingface_hub import hf_hub_download
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from fairseq_signals import tasks


def build_classifier(
    model_store: str,
    status: str = "pretrained",
    num_labels: int = 26,
    checkpoint: str = None,
):
    """
    Downloads model checkpoints and a configuration file from Hugging Face Hub,
    then initializes and builds a fairseq_signals ECG model from them.

    Args:
        model_store (str): Local directory to store model files.
        status (str): "pretrained" to download standard model, "custom" to load from checkpoint.
        checkpoint (str): Path to a custom checkpoint (required if status != "pretrained").

    Returns:
        model: The initialized ECG-FM model object.
    """
    # Clear Hydra if already initialized
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    model_store = Path(model_store).resolve()  # Hydra requires an absolute path
    checkpoint_folder = model_store / "ECG-FM"
    checkpoint_folder.mkdir(parents=True, exist_ok=True)

    if status == "pretrained":
        repo_id = "wanglab/ecg-fm"
        checkpoint_folder = checkpoint_folder / "pretrained"
        checkpoint_folder.mkdir(parents=True, exist_ok=True)

        cfg_file = checkpoint_folder / "physionet_finetuned.yaml"
        model_name = "mimic_iv_ecg_physionet_pretrained.pt"
        model_file = checkpoint_folder / model_name

        # fallback: download missing files
        if not cfg_file.exists():
            hf_hub_download(
                repo_id=repo_id,
                filename="physionet_finetuned.yaml",
                local_dir=str(checkpoint_folder),
                revision="8fb94b0",  # commit with config
            )

        if not model_file.exists():
            hf_hub_download(
                repo_id=repo_id,
                filename=model_name,
                local_dir=str(checkpoint_folder),
                revision="584219e",  # commit with weights
            )

        # Initialize Hydra with config
        initialize_config_dir(config_dir=str(checkpoint_folder))
        cfg = compose(
            config_name="physionet_finetuned.yaml",
            overrides=[
                f"model.num_labels={num_labels}",
                "+model.filter=False",
                f"model.model_path={str(model_file)}",
            ],
        )

        # Build model from task
        task = tasks.setup_task(cfg.task)
        model = task.build_model(cfg.model)

    elif status == 'finetuned':  # custom checkpoint
        if checkpoint is None:
            raise ValueError("You must provide a checkpoint path if status != 'pretrained'.")

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        model = build_model_from_checkpoint(checkpoint_path)

    else:
        raise ValueError("status must be 'pretrained' or 'finetuned'")
        
    return __reset_forward(model)


def __reset_forward(model):
    """
    Patches the fairseq_signals model so one can call model(X) instead of model(source=X)
    """
    original_forward = model.forward

    def new_forward(self, X, *args, **kwargs):
        return original_forward(source=X, *args, **kwargs)

    model.forward = new_forward.__get__(model, model.__class__)
    return model