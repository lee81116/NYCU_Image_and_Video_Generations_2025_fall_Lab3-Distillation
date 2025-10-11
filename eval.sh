#!/usr/bin/env bash
set -euo pipefail

# ---------------------------
# Unified evaluator for SDS / SDI / VSD
# Default: guidance = 25
# ---------------------------

LOSS=""
GUIDANCE=25
STEPS=500
DEVICE=0
NEGATIVE_PROMPT="low quality"
PROMPTS_FILE=""
SAVE_ROOT="./outputs"

# VSD hyperparams
LORA_LR=1e-4
LORA_LOSS_WEIGHT=1.0
LORA_RANK=4

usage() {
  echo "Usage: ./eval.sh (--sds | --sdi | --vsd) [--guidance <num>] [--steps <int>] [--device <int>] [--prompts-file <path>]"
  echo "Examples:"
  echo "  ./eval.sh --sds"
  echo "  ./eval.sh --sdi --guidance 15"
  echo "  ./eval.sh --vsd --guidance 7.5"
}

# ---------------------------
# Parse args
# ---------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sds) LOSS="sds"; shift ;;
    --sdi) LOSS="sdi"; shift ;;
    --vsd) LOSS="vsd"; shift ;;
    --guidance) GUIDANCE="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --prompts-file) PROMPTS_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$LOSS" ]]; then
  echo "[Error] You must specify one of --sds | --sdi | --vsd"
  usage
  exit 1
fi

SAVE_DIR="$SAVE_ROOT/$LOSS"
mkdir -p "$SAVE_DIR"

# ---------------------------
# Prompts
# ---------------------------
DEFAULT_PROMPTS=(
  "A red bus driving on a desert road"
  "a boat in a river"
  "A cabin surrounded by forests"
  "A church beside a lake"
  "A villa close to the pool"
  "A castle next to a river"
  "A burger on the table"
  "A dog sitting on grass"
  "a cat sitting on a table"
  "A car on the road"
)
if [[ -n "$PROMPTS_FILE" ]]; then
  mapfile -t PROMPTS < <(grep -v '^[[:space:]]*$' "$PROMPTS_FILE")
else
  PROMPTS=("${DEFAULT_PROMPTS[@]}")
fi

# ---------------------------
# Run generation
# ---------------------------
echo "[*] Running $LOSS | guidance=$GUIDANCE | steps=$STEPS | device=$DEVICE"
for prompt in "${PROMPTS[@]}"; do
  echo "=== Generating: $prompt ==="
  if [[ "$LOSS" == "vsd" ]]; then
    python main.py \
      --prompt "$prompt" \
      --negative_prompt "$NEGATIVE_PROMPT" \
      --loss_type "$LOSS" \
      --guidance_scale "$GUIDANCE" \
      --step "$STEPS" \
      --device "$DEVICE" \
      --save_dir "$SAVE_DIR" \
      --lora_lr "$LORA_LR" \
      --lora_loss_weight "$LORA_LOSS_WEIGHT" \
      --lora_rank "$LORA_RANK"
  else
    python main.py \
      --prompt "$prompt" \
      --negative_prompt "$NEGATIVE_PROMPT" \
      --loss_type "$LOSS" \
      --guidance_scale "$GUIDANCE" \
      --step "$STEPS" \
      --device "$DEVICE" \
      --save_dir "$SAVE_DIR"
  fi
  echo "---"
done

# ---------------------------
# Clean intermediate images
# ---------------------------
echo "[*] Cleaning intermediate step images in $SAVE_DIR"
find "$SAVE_DIR" -maxdepth 1 -type f -name '[0-9]*.png' -delete || true

# ---------------------------
# Evaluation
# ---------------------------
echo "[*] Running CLIP evaluation..."
python eval.py --fdir1 "$SAVE_DIR"
echo "[*] Done. Results at: $SAVE_DIR/eval.json"
