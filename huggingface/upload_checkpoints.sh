#!/bin/bash
# Upload the three fine-tuned checkpoints and their model cards to the Hugging Face Hub.
# Usage (from the repository root, after `hf auth login`):
#   bash huggingface/upload_checkpoints.sh            # creates private repositories
# Make the repositories public on the Hub once the model cards have been reviewed.
set -euo pipefail

ORG=${ORG:-AIBAT-LMU}
CKPT_DIR=${CKPT_DIR:-models/finetuned}
CARD_DIR=$(cd "$(dirname "$0")" && pwd)
HF=${HF:-uv run --frozen --no-sync hf}

upload() {
  local repo=$1 file=$2
  test -f "$CKPT_DIR/$file" || { echo "Missing checkpoint: $CKPT_DIR/$file"; exit 1; }
  echo "== $ORG/$repo"
  sha256sum "$CKPT_DIR/$file"
  $HF upload "$ORG/$repo" "$CARD_DIR/$repo/README.md" README.md --repo-type model --private
  $HF upload "$ORG/$repo" "$CKPT_DIR/$file" "$file" --repo-type model
}

upload ecg-fm-athletes-ecgfm            best_model_ecgfm.pth
upload ecg-fm-athletes-hubert-ecg-small best_model_hubertecgsmall.pth
upload ecg-fm-athletes-ecgfounder       best_model_ecgfounder.pth
