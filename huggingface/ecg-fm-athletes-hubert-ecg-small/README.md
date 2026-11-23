---
license: cc-by-nc-4.0
base_model: Edoardo-BS/hubert-ecg-small
library_name: pytorch
tags:
- ecg
- electrocardiography
- cardiology
- multi-label-classification
- sports-cardiology
---

# HuBERT-ECG (small) fine-tuned on PhysioNet/CinC 2021 (26 labels)

Fine-tuned weights of **HuBERT-ECG (small)** used in

> Goelz C, Lang S, Schlichtiger J, Brunner S, Vieluf S. ECG foundation models for cardiovascular
> screening in a cohort of young competitive athletes. *npj Digital Medicine* 9, 920 (2026).
> https://doi.org/10.1038/s41746-026-03474-9

Code to preprocess data, run the model and reproduce all results of the paper:
https://github.com/AIBAT-LMU/ecg-fm-athletes

The companion checkpoints are
[ECG-FM](https://huggingface.co/AIBAT-LMU/ecg-fm-athletes-ecgfm),
[HuBERT-ECG (small)](https://huggingface.co/AIBAT-LMU/ecg-fm-athletes-hubert-ecg-small) and
[ECGFounder](https://huggingface.co/AIBAT-LMU/ecg-fm-athletes-ecgfounder).

## Model

| | |
|---|---|
| Base model | [Edoardo-BS/hubert-ecg-small](https://huggingface.co/Edoardo-BS/hubert-ecg-small), revision `a6d3f2c` ([code](https://github.com/Edoar-do/HuBERT-ECG)) |
| Task | Multi-label classification of 26 diagnostic ECG labels |
| Input | 12 leads, 100 Hz, 5-second segments, band-pass filtering and rescaling as in the original HuBERT-ECG pipeline |
| Output | One logit per label; apply a sigmoid to obtain probabilities |
| File | `best_model_hubertecgsmall.pth` (PyTorch `state_dict`) |

Labels: AF, AFL, BBB, Brady, CLBBB|LBBB, CRBBB|RBBB, IAVB, IRBBB, LAD, LAnFB, LPR, LQRSV, LQT, NSIVCB, NSR, PAC|SVPB, PR, PRWP, PVC|VPB, QAb, RAD, SA, SB, STach, TAb, TInv.

The label definitions follow the scored labels of the PhysioNet/CinC Challenge 2021. The order of
the model outputs is defined by the preprocessing in the code repository; use the repository to
run the model rather than relying on the order in this list.

## Training

Fine-tuned on the PhysioNet/CinC Challenge 2021 training data (87,662 recordings after
preprocessing, the PTB and St. Petersburg INCART subsets excluded), split 80/10/10 into training,
validation and test sets with stratification by label.

- Partial fine-tuning: the convolutional feature extractor is frozen; the last eight transformer blocks and the classification head are trained
- AdamW (betas 0.9/0.98, eps 1e-9), initial learning rate 1e-5, weight decay 0.01
- Linear learning-rate schedule with 8% warm-up
- LayerDrop 0.1 and dropout 0.1
- `BCEWithLogitsLoss` with positive class weights, batch size 64
- Early stopping on the validation AUROC (patience 10); best model from epoch 28

The checkpoint is the one with the highest validation AUROC. The training script does not fix the
random seed, so re-running the fine-tuning gives a similar but not identical model. The results of
the paper are reproduced with exactly these weights.

## Evaluation

Macro-averaged metrics as reported in the paper. Threshold-dependent metrics use a fixed
threshold of 0.5.

| Dataset | AUROC | AUPRC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| PhysioNet/CinC 2021, test split | 0.94 | 0.48 | 0.81 | 0.93 | 0.42 |
| CODE-15% (six shared labels) | 0.97 | 0.49 | 0.87 | 0.94 | 0.41 |
| LMU athletes, n = 43 (95% CI) | 0.73 (0.61–0.84) | 0.57 (0.39–0.74) | 0.46 (0.23–0.71) | 0.93 (0.87–0.98) | 0.41 (0.18–0.63) |

The PhysioNet/CinC 2021 values are averaged over the labels that are also evaluable in the athlete
cohort. Performance on athletes is lower than on the clinical test data, in particular for
sensitivity; see the paper for the per-label results and the analysis of this domain shift.

## Usage

```bash
git clone https://github.com/AIBAT-LMU/ecg-fm-athletes
cd ecg-fm-athletes
uv sync
make download-models
```

```python
import torch
from huggingface_hub import hf_hub_download

from ecgath import models

checkpoint = hf_hub_download("AIBAT-LMU/ecg-fm-athletes-hubert-ecg-small", "best_model_hubertecgsmall.pth")
model = models.hubert_ecg.build_classifier(model_store="models/", size="small", num_labels=26)
model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
model.eval()
```

Inputs must be preprocessed as in the repository (`scripts/01_preprocess.py`).

## Intended use and limitations

- For research on automated ECG interpretation. **Not a medical device and not intended for
  clinical decision-making.**
- Trained on clinical ECGs of a general, mostly older population. The paper shows that
  performance drops on young competitive athletes, where findings such as sinus bradycardia or
  incomplete right bundle branch block are common and often physiological.
- The athlete cohort used for evaluation is small (43 male athletes from one centre); the reported
  intervals are wide.
- No athlete data were used for training. The athlete dataset is not public.

## License

Released under CC BY-NC 4.0, the license of the base model. **Non-commercial use only.** The training data are available under CC BY 4.0 from
[PhysioNet](https://doi.org/10.13026/34va-7q14).

## Citation

```bibtex
@article{goelz2026ecg,
  title   = {ECG foundation models for cardiovascular screening in a cohort of young competitive athletes},
  author  = {Goelz, Christian and Lang, Simon and Schlichtiger, Jenny and Brunner, Stefan and Vieluf, Solveig},
  journal = {npj Digital Medicine},
  volume  = {9},
  pages   = {920},
  year    = {2026},
  doi     = {10.1038/s41746-026-03474-9}
}
```

Please also cite the base model:

```bibtex
@article{coppola2026hubertecg,
  title   = {{HuBERT-ECG} as a self-supervised foundation model for broad and scalable cardiac applications},
  author  = {Coppola, E. and others},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.1101/2024.11.14.24317328},
  note    = {Preprint}
}
```

and the training data, the PhysioNet/CinC Challenge 2021:

```bibtex
@inproceedings{reyna2021willtwodo,
  title     = {Will two do? Varying dimensions in electrocardiography: the {PhysioNet/Computing in Cardiology Challenge} 2021},
  author    = {Reyna, M. A. and others},
  booktitle = {Computing in Cardiology},
  volume    = {48},
  pages     = {1--4},
  year      = {2021},
  doi       = {10.23919/cinc53138.2021.9662687}
}

@misc{reyna2022physionet,
  title     = {Will two do? Varying dimensions in electrocardiography: the {PhysioNet/Computing in Cardiology Challenge} 2021 (version 1.0.3)},
  author    = {Reyna, M. and others},
  publisher = {PhysioNet},
  year      = {2022},
  doi       = {10.13026/34va-7q14}
}

@article{goldberger2000physionet,
  title   = {{PhysioBank}, {PhysioToolkit}, and {PhysioNet}: components of a new research resource for complex physiologic signals},
  author  = {Goldberger, A. L. and others},
  journal = {Circulation},
  volume  = {101},
  pages   = {e215--e220},
  year    = {2000},
  doi     = {10.1161/01.cir.101.23.e215}
}
```
