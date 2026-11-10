# tinker-ai-skill

**Turn your own coding-agent sessions into a fine-tuned model you can run locally. This is the recipe behind Tess-4 and Tess-4.5.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Agent Skill](https://img.shields.io/badge/Agent%20Skill-agentskills.io-black)](https://agentskills.io)
[![Works with](https://img.shields.io/badge/works%20with-Claude%20Code%20%C2%B7%20Codex%20%C2%B7%20OpenCode-brightgreen)](#install-the-skill)
[![Trains on](https://img.shields.io/badge/trains%20on-Tinker-7b3fe4)](https://thinkingmachines.ai/tinker)

Every session you run with a coding agent is a worked example of how you want a
model to behave: your repositories, your conventions, your tools, your way of
debugging. This skill hands your agent the complete, production-tested pipeline to
turn those sessions into a supervised fine-tuning dataset, train a LoRA on
[Tinker](https://thinkingmachines.ai/tinker), merge it, and ship a GGUF that runs in
llama.cpp on your own machine. One afternoon, a few dollars to a few hundred, and a
model that works the way you do.

Install it once and say: *"fine-tune Qwen3.6-35B-A3B on my Claude Code sessions from
the last three months."* The agent takes it from there and stops before anything
that costs money.

## Proven on real releases

We built the Tess family of agentic coding models from our own Claude Code sessions
with exactly this method, and shipped them as open weights on Hugging Face:
[Tess-4-27B](https://huggingface.co/migtissera/Tess-4-27B)
([GGUF](https://huggingface.co/migtissera/Tess-4-27B-GGUF),
[EAGLE-3 draft](https://huggingface.co/migtissera/Tess-4-27B-EAGLE3)) and
[Tess-4.5-27B](https://huggingface.co/migtissera/Tess-4.5-27B)
([GGUF](https://huggingface.co/migtissera/Tess-4.5-27B-GGUF)). Read about them at
[continuityresearch.ai/tess-4-5](https://continuityresearch.ai/tess-4-5).

What is in this repository is the distilled, cleaned-up version of the pipeline that
produced them, from raw transcripts to a quantized GGUF, including the mistakes we
made so you do not have to.

## What you get

- **One `SKILL.md`** your agent follows: a nine-step workflow with the decision
  points spelled out and the rules that cost us runs.
- **Eleven small scripts**, plain Python and shell on top of tinker-cookbook and
  llama.cpp, each with an offline default and a `--help` that explains every flag.
- **Six reference notes**: the transcript format, the anonymization policy, the
  chunking and compaction design, Tinker training settings with the learning-rate
  evidence, export and serving, and the lessons learned.
- **A synthetic transcript generator** so you can dry-run the whole pipeline in
  five minutes without touching real data or spending a cent.

## The method, from transcripts to a local model

| Step | What happens | Script |
|---|---|---|
| 1. Collect | Copy Claude Code transcripts verbatim, with checksums; skip subagents and tiny sessions; exclude whole projects that must stay private | `collect_sessions.py` |
| 2. Clean | Normalize to OpenAI-style messages; drop harness noise; replace every credential by value; give each session its own random identity for usernames, folders, names, emails, orgs; fail if anything leaks | `normalize_sessions.py`, `anonymize_sessions.py` |
| 3. Slice | Cut sessions to the model's context length, measured with the model's real renderer, never splitting a tool call from its result | `build_dataset.py` |
| 4. Compact | Give every slice after the first a deterministic working-context block and a continuation bridge, the way a real harness compacts | `build_dataset.py` |
| 5. Split | Hold out whole sessions with a seeded hash so rebuilds and other models compare | `build_dataset.py` |
| 6. Train | LoRA on Tinker with the cookbook trainer, tool definitions on every row, W&B connected, offline smoke test first, cost shown before launch | `train_tinker.py` |
| 7. Check | Sample base vs checkpoints with identical rendering; judge behaviour, not loss | `sample_checkpoint.py` |
| 8. Merge | Download the LoRA from Tinker and merge it into the base model as a standalone bf16 HuggingFace model (or a PEFT adapter for vLLM/SGLang) | `export_checkpoint.py` |
| 9. Ship | Convert to GGUF, quantize, export the vision projector and draft head, and serve with llama.cpp using the training-matched template and sampling settings | `make_gguf.sh`, `serve_gguf.sh` |

## Why this and not a generic SFT tutorial

Agent transcripts are not chat data. The details below are the difference between a
model that works and one that looks fine on the loss curve and falls apart in use.

- **Per-session anonymization.** A single constant fake world in every row was
  memorized and later asserted over the real environment. Randomize per session.
- **Compaction context on continuation slices.** Long sessions outlive the context
  window; the model must learn to continue from a summary, not from nothing.
- **Tool definitions and the system prompt on every row.** Rows without them
  regressed tool use even though NLL improved.
- **Thinking kept in history.** Hybrid renderers silently strip earlier reasoning and
  train empty think blocks; the scripts rebuild the renderer to keep it.
- **Measure with the real tokenizer.** A tokenizer change moved rows by 14% and
  overflowed the budget. Rebuild per model.
- **Judge by behaviour.** Two "collapses" had monotonically falling loss. Sample.
- **Serve the way you trained.** A chat template that strips earlier thinking, or
  llama.cpp's default min-p, turned a good checkpoint into garble. The serve script
  carries the right flags.

## Quick start: five minutes on synthetic data

```
git clone https://github.com/drost-ai/tinker-ai-skill.git ~/.claude/skills/tinker-ai-skill
cd ~/.claude/skills/tinker-ai-skill
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt

python scripts/make_sample_transcript.py --out ./demo-claude-projects --turns 14 --long-results
python scripts/collect_sessions.py   --claude-dir ./demo-claude-projects --data-dir ./demo-data --min-bytes 1000
python scripts/normalize_sessions.py --data-dir ./demo-data
python scripts/anonymize_sessions.py --data-dir ./demo-data --name "Taylor Quintrell"
python scripts/build_dataset.py      --data-dir ./demo-data --model Qwen/Qwen3.6-35B-A3B --max-len 8192 \
                                     --system-prompt-file assets/system-prompt.md
python scripts/train_tinker.py       --data-dir ./demo-data --name qwen3.6-35b-a3b-8k   # offline smoke test
```

The last command prints a decoded training row, the token count and the resolved
configuration. Add `--train` to launch a real run; afterwards
`export_checkpoint.py` and `make_gguf.sh` take it to a GGUF you can run.

## Then on your own sessions

1. Raise Claude Code's transcript retention first: `"cleanupPeriodDays": 3650` in
   `~/.claude/settings.json`. The default deletes sessions after 30 days.
2. Copy `.env.example` to `.env` and add your `TINKER_API_KEY` and `WANDB_API_KEY`.
3. Write `identity.json` (see `identity.example.json`): your usernames, full name,
   emails, handles, company, domains.
4. Follow `SKILL.md`, or let your agent drive. The decisions that stay yours: which
   projects to exclude, which model and context length, the system prompt and tool
   definitions of the harness you will serve in, and the learning rate when Tinker
   has no calibrated default.

## Install the skill

Clone into your agent's skills directory. The directory name must be
`tinker-ai-skill`, which `git clone` gives you by default:

| Agent | Personal (all projects) | Project (this repo) |
|---|---|---|
| Claude Code | `~/.claude/skills/tinker-ai-skill/` | `.claude/skills/tinker-ai-skill/` |
| OpenAI Codex | `~/.agents/skills/tinker-ai-skill/` | `.agents/skills/tinker-ai-skill/` |
| OpenCode | `~/.config/opencode/skills/tinker-ai-skill/` | `.opencode/skills/tinker-ai-skill/` |

```
# Claude Code
git clone https://github.com/drost-ai/tinker-ai-skill.git ~/.claude/skills/tinker-ai-skill
# Codex
git clone https://github.com/drost-ai/tinker-ai-skill.git ~/.agents/skills/tinker-ai-skill
# OpenCode
git clone https://github.com/drost-ai/tinker-ai-skill.git ~/.config/opencode/skills/tinker-ai-skill
```

OpenCode also reads `.claude/skills` and `.agents/skills`, so one project copy serves
all three. Any other agent that supports the [Agent Skills](https://agentskills.io)
standard works the same way. Invoke it directly with `/tinker-ai-skill` in Claude
Code; Codex and OpenCode pick it up from its description.

## What it costs

Tinker bills per token. Training cost is train tokens times the model's train price
(see the [price list](https://tinker-docs.thinkingmachines.ai/tinker/models/models_and_pricing/));
the smoke test prints the token count before you launch. Ballpark figures at
November 2026 prices, one epoch of 64K rows:

| Dataset | Train tokens | Qwen3.6-35B-A3B | Inkling 2 Small | Qwen3.8-27B |
|---|---|---|---|---|
| 100 rows (a personal corpus) | ~5M | ~$6 | ~$8 | ~$21 |
| 1,000 rows (what our releases used) | ~45M | ~$55 | ~$70 | ~$185 |

Dense models cost more per token than MoE models of similar quality (Tinker prices
by active parameters); Qwen3.8-27B is the base of Tess-4.5.

Sampling and periodic evaluation are extra and small. The dataset steps are free
and run on your laptop.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/) (or pip)
- `tinker-cookbook>=0.6.0` (installs `tinker`, `torch`, `tml-renderers`) and `wandb`
- A [Tinker](https://tinker.thinkingmachines.ai) API key; a [Weights & Biases](https://wandb.ai/authorize) key is optional but recommended
- Claude Code transcripts under `~/.claude/projects/` (other agents: emit the message
  schema in `references/claude-code-transcripts.md` and continue from step 2)
- For steps 8-9: a built [llama.cpp](https://github.com/ggml-org/llama.cpp) checkout,
  and disk for the base model plus the merged copy

## Layout

```
SKILL.md                            instructions the agent follows
scripts/collect_sessions.py         step 1
scripts/normalize_sessions.py       step 2a
scripts/anonymize_sessions.py       step 2b
scripts/build_dataset.py            steps 3-5
scripts/train_tinker.py             step 6
scripts/sample_checkpoint.py        step 7
scripts/export_checkpoint.py        step 8: download the LoRA, merge, optional PEFT adapter
scripts/make_gguf.sh                step 9: BF16 GGUF, quantize, mmproj, MTP draft
scripts/serve_gguf.sh               step 9: llama-server with training-matched flags
scripts/make_sample_transcript.py   synthetic transcript for a dry run
scripts/common.py                   shared rendering (builder, trainer and sampler render identically)
references/                         format, anonymization, chunking, training, export and serving, lessons
assets/system-prompt.md             a generic agent system prompt to start from
identity.example.json, .env.example
```

## Data safety

`data/`, `demo-*/`, `runs/`, `export/`, `wandb/`, `.env` and `identity.json` are git-ignored. The anonymizer
fails rather than writing a file that still leaks. Review anonymized rows before
training, and never publish a dataset built from transcripts that belong to someone
else without their consent.

## Author and license

Written by [Migel Tissera](https://huggingface.co/migtissera), creator of the Tess
models. MIT license. Tinker and tinker-cookbook are products of Thinking Machines
Lab; this skill is an independent guide built on them.

If this saved you a failed training run, star the repository and share what you
trained.
