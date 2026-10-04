#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke run: one bundled video, one sample and only 20 denoising steps.
# Any argument below can be overridden after the config, for example:
#   bash inference_from_video.sh --num_steps 50 --num_test_instances 5
python3.10 inference_from_video.py \
  --config configs/inference_minimal.json \
  "$@"
