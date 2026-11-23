import torch
from torchmetrics import Metric

import torch
from torchmetrics import F1Score, Recall, Specificity, Precision, AUROC, AveragePrecision

from tqdm import tqdm


class MetricsManager:
    """
    Tracks multiple multilabel metrics and handles proper inputs:
    - Probabilities for AUROC/AUPRC
    - Thresholded predictions for F1, Recall, Precision, Specificity
    """
    def __init__(self, num_labels, device="cpu", threshold=0.5):
        self.device = device
        self.threshold = threshold

        self.metrics = {
            "f1-score": F1Score(task="multilabel", num_labels=num_labels, average=None).to(device),
            "recall": Recall(task="multilabel", num_labels=num_labels, average=None).to(device),
            "specificity": Specificity(task="multilabel", num_labels=num_labels, average=None).to(device),
            "precision": Precision(task="multilabel", num_labels=num_labels, average=None).to(device),
            "auroc": AUROC(task="multilabel", num_labels=num_labels, average=None).to(device),
            "auprc": AveragePrecision(task="multilabel", num_labels=num_labels, average=None).to(device),
        }

    def update(self, preds, targets):
        """
        preds: probabilities (sigmoid applied)
        targets: binary labels
        """
        preds, targets = preds.to(self.device), targets.to(self.device)

        for name, metric in self.metrics.items():
            if name in ["auroc", "auprc"]:
                metric.update(preds, targets)
            else:
                metric.update((preds > self.threshold).float(), targets)

    def compute(self):
        return {name: metric.compute().cpu() for name, metric in self.metrics.items()}

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()


class CinC2021(Metric):
    def __init__(self, weights, sinus_rhythm_index, num_classes):
        super().__init__()  # remove compute_on_step
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.sinus_rhythm_index = sinus_rhythm_index
        self.num_classes = num_classes
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("outputs", default=[], dist_reduce_fx="cat")
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.outputs.append(preds.float())
        self.labels.append(target.float())
    
    def compute(self):
        labels = torch.cat(self.labels, dim=0)
        outputs = torch.cat(self.outputs, dim=0)
        num_recordings, num_classes = labels.shape

        A =  _compute_modified_confusion_matrix(labels, outputs)
        observed_score = torch.sum(self.weights * A)

        A_correct = _compute_modified_confusion_matrix(labels, labels)
        correct_score = torch.sum(self.weights * A_correct)

        # inactive score
        inactive_outputs = torch.zeros((num_recordings, num_classes), dtype=torch.bool, device=device)
        inactive_outputs[:, self.sinus_rhythm_index] = True
        A_inactive = _compute_modified_confusion_matrix(labels, inactive_outputs)
        inactive_score = torch.sum(self.weights * A_inactive)

        if correct_score != inactive_score:
            normalized_score = (observed_score - inactive_score) / (correct_score - inactive_score)
        else:
            normalized_score = torch.tensor(0.0)

        return normalized_score

    import torch

    def _compute_modified_confusion_matrix(labels, outputs):
        """
        Compute a multi-label confusion matrix (labels vs. outputs) with partial credit.
        Args:
            labels:   (torch.Tensor, shape [num_recordings, num_classes], binary)
            outputs:  (torch.Tensor, shape [num_recordings, num_classes], binary)
        Returns:
            A:        (torch.Tensor, shape [num_classes, num_classes])
        """
        num_recordings, num_classes = labels.shape
        A = torch.zeros((num_classes, num_classes), dtype=torch.float32, device=labels.device)

        for i in range(num_recordings):
            # Compute normalization: number of active labels or outputs, at least 1
            normalization = float(
                max(
                    torch.sum(
                        torch.any(
                            torch.stack([labels[i].bool(), outputs[i].bool()]),
                            dim=0
                        )
                    ).item(),
                    1
                )
            )
            # Assign partial/full credit for each positive label/output
            for j in range(num_classes):
                if labels[i, j]:
                    for k in range(num_classes):
                        if outputs[i, k]:
                            A[j, k] += 1.0 / normalization

        return A
