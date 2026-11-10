#!/usr/bin/env bash
# Step 9b: serve a GGUF with llama-server using the training-matched settings.
#
# What matters for a model trained with this skill:
#   --jinja                   use the model's chat template (tool calls, thinking)
#   --reasoning-preserve      keep earlier-turn thinking in the prompt, as training did
#   --chat-template-kwargs    the template's own switch for the same thing
#                             (Qwen3.5 family: {"preserve_thinking": true};
#                              Nemotron: {"truncate_history_thinking": false})
#   -c                        context at least the training window (64K by default)
#   temperature 1.0, top-p 0.95, top-k 20, min-p 0   the base model's sampling defaults;
#                             llama.cpp's default min-p of 0.05 is not what training saw
#   KV cache q8_0 if you must quantize it; q4_0 caused reasoning loops past ~90K tokens
#
# Usage: scripts/serve_gguf.sh <model.gguf>
# Environment: CTX (65536) PORT (8080) HOST (127.0.0.1) NGL (99) MMPROJ (path) MTP (draft gguf path)
#              TEMPLATE_KWARGS ('{"preserve_thinking": true}') KV (q8_0 | f16) LLAMA_CPP (~/src/llama.cpp)
set -euo pipefail
case "${1:-}" in -h|--help) sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;; esac
MODEL="${1:?model.gguf}"
LLAMA_CPP="${LLAMA_CPP:-$HOME/src/llama.cpp}"
# Kept in a variable: a literal "}" inside ${VAR:-default} ends the expansion early,
# which appended a stray "}" to any TEMPLATE_KWARGS the caller set.
DEFAULT_TEMPLATE_KWARGS='{"preserve_thinking": true}'
SERVER="$LLAMA_CPP/build/bin/llama-server"; [ -x "$SERVER" ] || SERVER="$(command -v llama-server || true)"
[ -n "$SERVER" ] || { echo "llama-server not found (set LLAMA_CPP)"; exit 1; }

ARGS=(-m "$MODEL" -ngl "${NGL:-99}" -c "${CTX:-65536}" -fa on --jinja --reasoning-preserve
      --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
      --chat-template-kwargs "${TEMPLATE_KWARGS:-$DEFAULT_TEMPLATE_KWARGS}"
      --port "${PORT:-8080}" --host "${HOST:-127.0.0.1}")
[ -n "${KV:-}" ] && ARGS+=(-ctk "$KV" -ctv "$KV")
[ -n "${MMPROJ:-}" ] && ARGS+=(--mmproj "$MMPROJ")
[ -n "${MTP:-}" ] && ARGS+=(-md "$MTP" --spec-type draft-mtp -ngld 99)
echo "+ $SERVER ${ARGS[*]}"
exec "$SERVER" "${ARGS[@]}"
