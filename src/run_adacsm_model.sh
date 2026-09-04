#!/bin/bash

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# -----------------------------------------------------------------------------
# Editable defaults for a single AdaCSM run.
# You can modify these directly, or override from CLI.
# Example:
#   bash src/run_adacsm_model.sh --dataset PBC --learning_rate 0.000468 --top_k 2
# -----------------------------------------------------------------------------
DATASET="${DATASET:-FRAMINGHAM}"
LEARNING_RATE="${LEARNING_RATE:-0.00621}"
DISCOUNT="${DISCOUNT:-0.5659}"
LAYERS="${LAYERS:-[100]}"
NUM_EXPERTS="${NUM_EXPERTS:-32}"
TOP_K="${TOP_K:-2}"
MOE_DROPOUT="${MOE_DROPOUT:-0.0544}"
GATE_DROPOUT="${GATE_DROPOUT:-0.0982}"
GATE_TEMPERATURE="${GATE_TEMPERATURE:-0.8553}"
LOAD_BALANCE_LAMBDA="${LOAD_BALANCE_LAMBDA:-0.0725}"
ITERS="${ITERS:-2000}"
PATIENCE="${PATIENCE:-200}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export ADACSM_NO_SHOW="${ADACSM_NO_SHOW:-1}"

if [[ " $* " != *" --dataset "* ]]; then
  set -- --dataset "$DATASET" "$@"
fi
if [[ " $* " != *" --learning_rate "* ]]; then
  set -- --learning_rate "$LEARNING_RATE" "$@"
fi
if [[ " $* " != *" --discount "* ]]; then
  set -- --discount "$DISCOUNT" "$@"
fi
if [[ " $* " != *" --layers "* ]]; then
  set -- --layers "$LAYERS" "$@"
fi
if [[ " $* " != *" --num_experts "* ]]; then
  set -- --num_experts "$NUM_EXPERTS" "$@"
fi
if [[ " $* " != *" --top_k "* ]]; then
  set -- --top_k "$TOP_K" "$@"
fi
if [[ " $* " != *" --moe_dropout "* ]]; then
  set -- --moe_dropout "$MOE_DROPOUT" "$@"
fi
if [[ " $* " != *" --gate_dropout "* ]]; then
  set -- --gate_dropout "$GATE_DROPOUT" "$@"
fi
if [[ " $* " != *" --gate_temperature "* ]]; then
  set -- --gate_temperature "$GATE_TEMPERATURE" "$@"
fi
if [[ " $* " != *" --load_balance_lambda "* ]]; then
  set -- --load_balance_lambda "$LOAD_BALANCE_LAMBDA" "$@"
fi
if [[ " $* " != *" --iters "* ]]; then
  set -- --iters "$ITERS" "$@"
fi
if [[ " $* " != *" --patience "* ]]; then
  set -- --patience "$PATIENCE" "$@"
fi

exec python main.py "$@"
