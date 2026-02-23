#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <checkpoint_path> <dataset: webqsp|cwq> [out_dir]"
  exit 1
fi

ckpt_path="$1"
dataset="$2"
out_dir="${3:-$(dirname "$ckpt_path")}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for b in 20 40 60; do
  out_pth="${out_dir}/retrieval_result_backbone_b${b}.pth"
  echo "[Backbone grid] Running backbone_budget=${b}, output=${out_pth}"
  python "${script_dir}/inference_backbone.py" \
    -p "${ckpt_path}" \
    -d "${dataset}" \
    --backbone_budget "${b}" \
    --output_path "${out_pth}"

  eval_txt="${out_pth%.pth}.eval.txt"
  python "${script_dir}/eval.py" -d "${dataset}" -p "${out_pth}" --k_list 50,100,200,400,500 | tee "${eval_txt}"
done
