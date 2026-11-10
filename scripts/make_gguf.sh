#!/usr/bin/env bash
# Step 9a: merged HuggingFace model -> BF16 GGUF -> quantized GGUFs (llama.cpp).
#
# Requires a llama.cpp checkout that is recent enough for the architecture and
# built (cmake -B build && cmake --build build --config Release -j), plus the
# converter's Python deps (pip install -r $LLAMA_CPP/requirements/requirements-convert_hf_to_gguf.txt
# into the same environment as this skill, or rely on gguf-py via PYTHONPATH as below).
# Package builds of llama.cpp often lag new architectures; use the checkout's
# build/bin/llama-quantize from the SAME commit as convert_hf_to_gguf.py.
#
# Usage: scripts/make_gguf.sh <merged_hf_dir> <out_dir> <model_name> [QUANT ...]
#   QUANT defaults to Q4_K_M. Q8_0 and Q6_K are near-lossless; Q4_K_M is the usual
#   size/quality trade-off for local use.
# Environment:
#   LLAMA_CPP   path to the llama.cpp checkout (default: ~/src/llama.cpp)
#   PYTHON      interpreter with the converter deps (default: python3)
#   OUTTYPE     bf16 (default) or f16 for the full-precision GGUF
#   MMPROJ=0    skip the vision projector even if the model has a vision tower
#   MTP=1       also export a multi-token-prediction draft GGUF (Qwen3.5-family
#               models ship an MTP head; serve it with --spec-type draft-mtp)
#   SPLIT_MAX_SIZE=45G  split outputs for HuggingFace's 50 GB file limit
set -euo pipefail
case "${1:-}" in -h|--help) sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;; esac

MERGED="${1:?merged HF model dir}"
OUT="${2:?output dir}"
NAME="${3:?model name, e.g. my-coder-27b}"
shift 3
if [ $# -eq 0 ]; then QUANTS=(Q4_K_M); else QUANTS=("$@"); fi

LLAMA_CPP="${LLAMA_CPP:-$HOME/src/llama.cpp}"
PYTHON="${PYTHON:-python3}"
OUTTYPE="${OUTTYPE:-bf16}"
CONVERT="$LLAMA_CPP/convert_hf_to_gguf.py"
QUANT="$LLAMA_CPP/build/bin/llama-quantize"
[ -f "$CONVERT" ] || { echo "missing $CONVERT (set LLAMA_CPP)"; exit 1; }
[ -x "$QUANT" ]   || { echo "missing $QUANT (build llama.cpp: cmake -B build && cmake --build build --config Release -j)"; exit 1; }
[ -f "$MERGED/config.json" ] || { echo "missing $MERGED/config.json"; exit 1; }
mkdir -p "$OUT"

SPLIT=(); [ -n "${SPLIT_MAX_SIZE:-}" ] && SPLIT=(--split-max-size "$SPLIT_MAX_SIZE")
HAS_MTP=0; grep -q -E '"num_nextn_predict_layers"|"mtp' "$MERGED/config.json" && HAS_MTP=1
HAS_VISION=0; grep -q '"vision_config"' "$MERGED/config.json" && HAS_VISION=1
NO_MTP=(); [ "$HAS_MTP" = 1 ] && NO_MTP=(--no-mtp)

FULL="$OUT/${NAME}-$(echo "$OUTTYPE" | tr '[:lower:]' '[:upper:]').gguf"
echo ">> convert HF -> $OUTTYPE GGUF: $FULL"
# ${ARR[@]+"${ARR[@]}"}: an empty array under `set -u` is an "unbound variable" error
# before bash 4.4, and macOS still ships bash 3.2 as /bin/bash.
PYTHONPATH="$LLAMA_CPP/gguf-py" "$PYTHON" "$CONVERT" "$MERGED" --outfile "$FULL" --outtype "$OUTTYPE" \
  --model-name "$NAME" ${NO_MTP[@]+"${NO_MTP[@]}"} ${SPLIT[@]+"${SPLIT[@]}"}

for q in "${QUANTS[@]}"; do
  echo ">> quantize -> $q"
  KEEP=(); [ -n "${SPLIT_MAX_SIZE:-}" ] && KEEP=(--keep-split)
  "$QUANT" ${KEEP[@]+"${KEEP[@]}"} "$FULL" "$OUT/${NAME}-${q}.gguf" "$q"
done

if [ "$HAS_VISION" = 1 ] && [ "${MMPROJ:-1}" != 0 ]; then
  echo ">> export vision projector (mmproj)"
  PYTHONPATH="$LLAMA_CPP/gguf-py" "$PYTHON" "$CONVERT" "$MERGED" --outfile "$OUT/mmproj-${NAME}-F16.gguf" \
    --outtype f16 --mmproj --model-name "$NAME"
fi

if [ "$HAS_MTP" = 1 ] && [ "${MTP:-0}" = 1 ]; then
  echo ">> export MTP draft head (speculative decoding)"
  PYTHONPATH="$LLAMA_CPP/gguf-py" "$PYTHON" "$CONVERT" "$MERGED" --outfile "$OUT/mtp-${NAME}-BF16.gguf" \
    --outtype bf16 --mtp --model-name "$NAME"
  "$QUANT" "$OUT/mtp-${NAME}-BF16.gguf" "$OUT/mtp-${NAME}-Q8_0.gguf" Q8_0
fi

echo ">> outputs:"; ls -la "$OUT"
echo ">> next: scripts/serve_gguf.sh $OUT/${NAME}-${QUANTS[0]}.gguf"
