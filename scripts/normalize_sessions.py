#!/usr/bin/env python3
"""Step 2a: turn raw Claude Code transcripts into clean conversation files.

Input:  <data-dir>/originals/**/*.jsonl (from collect_sessions.py)
Output: <data-dir>/normalized/<session-id>.jsonl -- a header record followed by one
        OpenAI-style message per line -- plus manifest.json.

What is kept: user text the person typed, assistant text, assistant thinking when
the transcript stores it in plaintext, tool calls (name + JSON arguments, id) and
tool results (head + tail of long outputs). What is dropped: subagent sidechain
records, meta/harness-injected user text (<system-reminder>, <command-...>,
task notifications), API-error placeholder messages, thinking signatures, image
bytes (replaced by "[image omitted]"), token accounting and every runtime record
type that is not a user/assistant message.

Message schema (one per line after the header):
  {"role": "user", "content": "..."}
  {"role": "assistant", "content": [{"type": "thinking", "thinking": "..."},
                                    {"type": "text", "text": "..."}],
   "tool_calls": [{"id": "...", "type": "function",
                   "function": {"name": "Read", "arguments": "{...json...}"}}]}
  {"role": "tool", "content": "...", "tool_call_id": "..."}

Paths, names and other personal context are untouched here on purpose: the
anonymizer rewrites them consistently in the next step.

Usage:
    python scripts/normalize_sessions.py --data-dir ./data
    python scripts/normalize_sessions.py --data-dir ./data --tool-result-head 8000 --tool-result-tail 4000
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import clip_middle, dump, is_harness_text, sha256_file, write_json, write_jsonl  # noqa: E402

IMAGE_BLOCKS = {"image", "input_image", "output_image", "image_url"}


def block_text(content):
    """Text of a tool_result / message content that may be a string or a list of blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            parts.append(b.get("text") or "")
        elif b.get("type") in IMAGE_BLOCKS:
            parts.append("[image omitted]")
    return "\n".join(p for p in parts if p)


def record_events(record, stats):
    """Yield normalized events for one transcript record (or nothing)."""
    kind = record.get("type")
    stats["records"] += 1
    if kind not in ("user", "assistant"):
        stats["runtime_records_skipped"] += 1
        return
    if record.get("isSidechain"):
        stats["sidechain_records_skipped"] += 1
        return
    if kind == "user" and record.get("isMeta"):
        stats["meta_user_records_skipped"] += 1
        return
    if kind == "assistant" and record.get("isApiErrorMessage"):
        stats["api_error_records_skipped"] += 1
        return
    content = (record.get("message") or {}).get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return
    uid = record.get("uuid")
    # Claude Code writes one record per content block; the blocks of one model
    # response share message.id, which is how late parallel tool calls are regrouped.
    mid = (record.get("message") or {}).get("id") if kind == "assistant" else None
    for index, block in enumerate(content):
        if not isinstance(block, dict):
            continue
        bt = block.get("type")
        key = (uid, index)
        if bt == "text":
            text = (block.get("text") or "").strip()
            if not text:
                continue
            if kind == "user" and is_harness_text(text):
                stats["harness_text_skipped"] += 1
                continue
            yield {"role": kind, "text": text, "key": key}
        elif bt == "thinking":
            text = (block.get("thinking") or "").strip()
            if block.get("signature"):
                stats["thinking_signatures_dropped"] += 1
            if text:
                yield {"role": "thinking", "text": text, "key": key}
            else:
                stats["empty_thinking_blocks"] += 1
        elif bt == "redacted_thinking":
            stats["redacted_thinking_skipped"] += 1
        elif bt == "tool_use":
            if kind != "assistant" or not isinstance(block.get("id"), str):
                stats["malformed_tool_use_skipped"] += 1
                continue
            yield {"role": "tool_call", "name": block.get("name") or "unknown",
                   "arguments": block.get("input") if isinstance(block.get("input"), dict) else {"input": block.get("input")},
                   "id": block["id"], "key": key, "mid": mid}
        elif bt == "tool_result":
            call_id = block.get("tool_use_id")
            if not isinstance(call_id, str):
                stats["malformed_tool_result_skipped"] += 1
                continue
            text = block_text(block.get("content"))
            if block.get("is_error"):
                text = "[tool error]\n" + text
            yield {"role": "tool_result", "text": text, "id": call_id, "key": key}
        elif bt in IMAGE_BLOCKS:
            stats["images_omitted"] += 1
            yield {"role": kind, "text": "[image omitted]", "key": key}
        else:
            stats[f"block_skipped:{bt}"] += 1


def pair_tool_events(events, stats):
    """Drop tool calls without a recorded result and results without a call."""
    results = {e["id"] for e in events if e["role"] == "tool_result"}
    calls = {e["id"] for e in events if e["role"] == "tool_call"}
    kept = []
    for e in events:
        if e["role"] == "tool_call" and e["id"] not in results:
            stats["unpaired_tool_calls_dropped"] += 1
            continue
        if e["role"] == "tool_result" and e["id"] not in calls:
            stats["unpaired_tool_results_dropped"] += 1
            continue
        kept.append(e)
    return kept


def to_messages(events, head, tail, stats):
    """Event stream -> messages. One assistant message per step: a step is the run
    of thinking / text / tool-call events between observations or user turns.

    Parallel tool calls need care: results are logged as each tool finishes, so a
    fast tool's result can precede a later tool_use block of the SAME response
    (call A, call B, result B, call C, result C, result A). Such a late call is
    attached to the assistant message already emitted instead of opening a new
    step, which would leave call A looking unanswered."""
    msgs, cur = [], None
    last = None        # assistant message emitted last: {"msg", "mid", "pending" call ids}

    def flush():
        nonlocal cur, last
        if cur is None:
            return
        parts = []
        if cur["thinking"]:
            parts.append({"type": "thinking", "thinking": "\n\n".join(cur["thinking"])})
        if cur["text"]:
            parts.append({"type": "text", "text": "\n\n".join(cur["text"])})
        if parts or cur["tool_calls"]:
            m = {"role": "assistant", "content": parts}
            if cur["tool_calls"]:
                m["tool_calls"] = cur["tool_calls"]
            msgs.append(m)
            last = {"msg": m, "mid": cur["mid"], "pending": {tc["id"] for tc in cur["tool_calls"]}}
        cur = None

    def open_step():
        nonlocal cur
        if cur is None:
            cur = {"thinking": [], "text": [], "tool_calls": [], "mid": None}

    for e in events:
        role = e["role"]
        if role in ("thinking", "assistant"):
            if cur and cur["tool_calls"]:
                flush()                      # text after tool calls starts a new step
            open_step()
            cur["thinking" if role == "thinking" else "text"].append(e["text"])
        elif role == "tool_call":
            call = {"id": e["id"], "type": "function",
                    "function": {"name": e["name"], "arguments": json.dumps(e["arguments"], ensure_ascii=False)}}
            same_response = bool(e.get("mid")) and last is not None and e["mid"] == last["mid"]
            if cur is None and last is not None and last["msg"].get("tool_calls") and (last["pending"] or same_response):
                last["msg"]["tool_calls"].append(call)      # late block of the same response
                last["pending"].add(e["id"])
                stats["late_parallel_tool_calls_regrouped"] += 1
            else:
                open_step()
                cur["tool_calls"].append(call)
                cur["mid"] = e.get("mid")
        elif role == "tool_result":
            flush()
            if last is not None:
                last["pending"].discard(e["id"])
            text = e["text"]
            if len(text) > head + tail + 200:
                stats["tool_results_clipped"] += 1
            msgs.append({"role": "tool", "content": clip_middle(text, head, tail), "tool_call_id": e["id"]})
        elif role == "user":
            flush()
            last = None                      # a user turn closes the assistant turn
            text = e["text"]
            if text == "[image omitted]" and (not msgs or msgs[-1]["role"] != "user"):
                stats["image_only_user_messages_dropped"] += 1
                continue
            if msgs and msgs[-1]["role"] == "user":
                msgs[-1]["content"] += "\n\n" + text   # two adjacent user texts are one turn
            else:
                msgs.append({"role": "user", "content": text})
    flush()
    return msgs


def check_structure(msgs):
    """Every tool call must be answered by its tool message before the next assistant turn."""
    pending = set()
    for m in msgs:
        if m["role"] == "assistant":
            if pending:
                raise ValueError(f"tool calls {sorted(pending)[:3]} never answered")
            pending = {tc["id"] for tc in m.get("tool_calls", [])}
        elif m["role"] == "tool":
            if m["tool_call_id"] not in pending:
                raise ValueError(f"tool result {m['tool_call_id']} without a preceding call")
            pending.discard(m["tool_call_id"])
        elif m["role"] == "user" and pending:
            raise ValueError("user turn while tool calls are unanswered")
    if pending:
        raise ValueError("conversation ends with unanswered tool calls")


def normalize_file(path, head, tail):
    stats = Counter()
    events, seen = [], set()
    cwd, models, first, last, sid = Counter(), Counter(), None, None, None
    with open(path, "rb") as f:
        for raw in f:
            try:
                r = json.loads(raw)
            except ValueError:
                stats["unparseable_lines"] += 1
                continue
            if not isinstance(r, dict):
                continue
            sid = sid or r.get("sessionId")
            if r.get("type") in ("user", "assistant") and not r.get("isSidechain"):
                if r.get("cwd"):
                    cwd[r["cwd"]] += 1
                ts = r.get("timestamp")
                if ts:
                    first, last = min(first or ts, ts), max(last or ts, ts)
                model = (r.get("message") or {}).get("model")
                if r.get("type") == "assistant" and model and model != "<synthetic>":
                    models[model] += 1
            for e in record_events(r, stats):
                if e["key"] in seen:              # the same block logged twice
                    stats["duplicate_blocks_skipped"] += 1
                    continue
                seen.add(e["key"])
                events.append(e)
    events = pair_tool_events(events, stats)
    msgs = to_messages(events, head, tail, stats)
    # Trailing tool messages with no assistant turn after them are still valid; a
    # leading tool message is not.
    while msgs and msgs[0]["role"] != "user":
        stats["leading_non_user_messages_dropped"] += 1
        msgs.pop(0)
    check_structure(msgs)
    header = {"type": "session", "session_id": sid or path.stem, "source_file": str(path),
              "source_sha256": sha256_file(path), "cwd": cwd.most_common(1)[0][0] if cwd else None,
              "started_at": first, "ended_at": last, "models": dict(models),
              "messages": len(msgs), "assistant_messages": sum(1 for m in msgs if m["role"] == "assistant"),
              "tool_calls": Counter(tc["function"]["name"] for m in msgs for tc in m.get("tool_calls", [])),
              "normalization": dict(stats)}
    return header, msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--tool-result-head", type=int, default=8000, help="characters kept at the start of a long tool result")
    ap.add_argument("--tool-result-tail", type=int, default=4000, help="characters kept at the end of a long tool result")
    ap.add_argument("--min-assistant", type=int, default=5)
    args = ap.parse_args()

    src = args.data_dir / "originals"
    out = args.data_dir / "normalized"
    files = sorted(src.glob("*/*.jsonl"))
    if not files:
        sys.exit(f"no transcripts under {src}; run collect_sessions.py first")
    out.mkdir(parents=True, exist_ok=True)
    sessions, skipped, totals = [], [], Counter()
    for path in files:
        try:
            header, msgs = normalize_file(path, args.tool_result_head, args.tool_result_tail)
        except ValueError as e:
            skipped.append({"file": str(path), "reason": f"structure: {e}"})
            print(f"skip {path.name}: {e}", file=sys.stderr)
            continue
        if header["assistant_messages"] < args.min_assistant:
            skipped.append({"file": str(path), "reason": "too_few_assistant_messages",
                            "assistant_messages": header["assistant_messages"]})
            continue
        target = out / f"{header['session_id']}.jsonl"
        write_jsonl(target, [header] + msgs)
        totals.update(header["normalization"])
        sessions.append({**{k: header[k] for k in ("session_id", "cwd", "started_at", "ended_at", "models",
                                                    "messages", "assistant_messages")},
                         "file": target.name, "sha256": sha256_file(target),
                         "tool_calls": sum(header["tool_calls"].values())})
        print(f"{header['session_id']}: {header['messages']} messages, {header['assistant_messages']} assistant, "
              f"{sum(header['tool_calls'].values())} tool calls", flush=True)
    write_json(out / "manifest.json", {"sessions": sessions, "skipped": skipped, "totals": dict(totals),
                                       "tool_result_policy": {"head": args.tool_result_head, "tail": args.tool_result_tail}})
    print(dump({"sessions": len(sessions), "skipped": len(skipped), "totals": dict(totals)}))
    if not sessions:
        sys.exit("no sessions kept")


if __name__ == "__main__":
    main()
