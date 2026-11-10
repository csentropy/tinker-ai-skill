#!/usr/bin/env python3
"""Steps 3-5: slice sessions to the context length, add compaction context to every
slice after the first, and write train/val files.

Input:  <data-dir>/anonymized/*.jsonl      Output: <data-dir>/datasets/<name>.{train,val}.jsonl
                                                   + <name>.manifest.json + <name>.report.json

How a session becomes rows:
  * Token budget is measured with the REAL renderer and tokenizer of --model (the same
    code path the trainer uses), including the system prompt and the tool declaration.
  * Boundaries are "safe": a window never separates a tool call from its results and
    never opens on a tool message. Windows are packed from a raw token estimate, then
    every window is rendered; if one overflows, the whole session is re-packed with a
    smaller window until everything fits.
  * Slice 0 gets the system prompt (if configured) with the working directory.
  * Every later slice gets the same system prompt followed by a deterministic
    <working_context> block that summarizes what came before: the original request,
    recent requests, files touched, recent commands, task state, and a near-raw
    trace of the last few messages. If the window opens mid-turn (on an assistant
    message) a short "continuation" user message bridges into it. This is what a
    coding harness does when it compacts a long session, so the model learns to
    continue from compacted context instead of from nothing.
  * Windows that would contain no assistant tokens are merged into a neighbour; the
    rows of one session cover its messages exactly once.
  * Tool definitions are declared on every row (a model trained without them regressed
    on tool use). Pass --tools-file with the real OpenAI-style definitions of your
    harness; otherwise minimal schemas are inferred from the arguments observed.
  * train/val split is by WHOLE session, chosen by a seeded hash so rebuilds and
    other model profiles hold out the same sessions.

Usage:
    python scripts/build_dataset.py --data-dir ./data --model Qwen/Qwen3.6-35B-A3B --max-len 65536 \
        --system-prompt-file assets/system-prompt.md
    python scripts/build_dataset.py --data-dir ./data --model thinkingmachines/Inkling-2-Small --max-len 65536 --effort 0.9
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (dump, make_renderer, model_slug, read_session, sha256_file,  # noqa: E402
                    token_count_and_loss, write_json, write_jsonl)

MSG_OVERHEAD = 12          # per-message framing estimate for packing; exact sizes come from the renderer
SYS_MARGIN = 256
PRESERVED_TAIL_MSGS = 4
SPLIT_SEED = "agent-transcript-session-split-v1"
WC_PREAMBLE = ("Archived working context from earlier turns and compacted portions of the current turn is "
               "available below.\nUse it as authoritative background for earlier work and earlier compacted "
               "activity in this turn.\nWhen it conflicts with the recent raw turns, prefer the recent raw turns.")
BRIDGE = ("Continuation context: earlier history and older parts of the current tool-driven turn were compacted "
          "into the working context provided in the system prompt. Continue from the recent raw messages below.")
ENV_BLOCK = "\n\n# Environment\n\nWorking directory: {cwd}"
PATH_KEYS = ("file_path", "path", "notebook_path", "filename", "target_file")
COMMAND_KEYS = ("command", "cmd")


# ------------------------------------------------------------ token estimates
class Budget:
    def __init__(self, tokenizer, renderer, provenance, effort, max_len):
        self.tok, self.renderer, self.prov, self.effort, self.max_len = tokenizer, renderer, provenance, effort, max_len
        self.raw_window = int(max_len * 0.82)
        self.shrink = max(512, max_len // 16)
        self.floor = max(1024, int(max_len * 0.35))
        self.wc_cap = max(512, min(4000, max_len // 16))

    def text_tokens(self, text):
        return len(self.tok.encode(text))

    def est_message(self, m):
        parts = []
        if isinstance(m.get("content"), str):
            parts.append(m["content"])
        for p in m.get("content") or [] if isinstance(m.get("content"), list) else []:
            parts.append(p.get("thinking") or p.get("text") or "")
        for tc in m.get("tool_calls") or []:
            parts.append(tc["function"]["name"] + "\n" + tc["function"]["arguments"])
        return self.text_tokens("\n".join(parts)) + MSG_OVERHEAD

    def size(self, messages, tools):
        return token_count_and_loss(self.renderer, messages, tools, self.effort, self.prov)


# ------------------------------------------------------- boundaries + packing
def safe_boundaries(M):
    safe, open_ids = {0, len(M)}, set()
    for j, m in enumerate(M):
        if m["role"] == "assistant":
            open_ids.update(tc["id"] for tc in m.get("tool_calls") or [])
        elif m["role"] == "tool":
            open_ids.discard(m.get("tool_call_id"))
        nxt = M[j + 1] if j + 1 < len(M) else None
        if not open_ids and (nxt is None or nxt["role"] != "tool"):
            safe.add(j + 1)
    return sorted(safe)


def pack(M, pref, window):
    bnds = safe_boundaries(M)
    chunks, start = [], 0
    while start < len(M):
        best = None
        for b in bnds:
            if b <= start:
                continue
            if pref[b] - pref[start] <= window:
                best = b
            else:
                break
        if best is None:                        # a single exchange larger than the window
            best = min(b for b in bnds if b > start)
        chunks.append((start, best))
        start = best
    return chunks


def exact_cover(ranges, n):
    cursor = 0
    for a, b in ranges:
        if a != cursor or b <= a:
            return False
        cursor = b
    return cursor == n


# ---------------------------------------------------------- working context
def _args(tc):
    try:
        return json.loads(tc["function"]["arguments"])
    except Exception:
        return {}


def safe_clip(value, limit):
    text = str(value)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if cut and not cut[-1].isspace() and not text[limit].isspace():
        boundary = max(cut.rfind(" "), cut.rfind("\n"), cut.rfind("\t"))
        if boundary >= 0:
            cut = cut[:boundary]
    return cut.rstrip() + " …"


def message_text(m):
    if isinstance(m.get("content"), str):
        return m["content"]
    return "\n".join(p.get("text", "") for p in m.get("content") or [] if p.get("type") == "text")


def build_working_context(prefix, session_id, generation, cwd, budget):
    users = [m["content"] for m in prefix if m["role"] == "user" and isinstance(m.get("content"), str)]
    files, commands, tasks = OrderedDict(), [], OrderedDict()
    for m in prefix:
        if m["role"] != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            name, a = tc["function"]["name"], _args(tc)
            path = next((a[k] for k in PATH_KEYS if isinstance(a.get(k), str)), None)
            if path:
                files[path] = "modified" if re.search(r"edit|write|patch|create", name, re.I) else files.get(path, "read")
            cmd = next((a[k] for k in COMMAND_KEYS if isinstance(a.get(k), str)), None)
            if cmd:
                commands.append(cmd.replace("\n", " "))
            if name == "TaskCreate":
                tasks[str(len(tasks) + 1)] = {"subject": a.get("subject", ""), "activeForm": a.get("activeForm", ""),
                                              "status": "pending"}
            elif name == "TaskUpdate" and a.get("taskId") in tasks and a.get("status"):
                tasks[a["taskId"]]["status"] = a["status"]
    open_todos = [t["subject"] for t in tasks.values() if t["status"] != "completed"]
    completed = [t["subject"] for t in tasks.values() if t["status"] == "completed"]
    in_progress = [t["activeForm"] or t["subject"] for t in tasks.values() if t["status"] == "in_progress"]

    L = ["<working_context>", f"session={session_id}", f"generation={generation}", f"cwd={cwd or '(unknown)'}"]

    def section(head, items, n=None, trunc=200):
        if items:
            L.append(f"{head}:")
            L.extend(f"- {safe_clip(str(it).strip(), trunc)}" for it in (items[-n:] if n else items))

    section("active_goals", users[:1], trunc=400)
    section("recent_requests", users[1:], n=3, trunc=240)
    section("relevant_files", [f"{p} ({r})" for p, r in files.items()], n=15, trunc=160)
    section("recent_commands", commands, n=8, trunc=120)
    section("in_progress", in_progress, n=6)
    section("open_todos", open_todos, n=12)
    section("completed", completed, n=12)
    section("next_steps", (in_progress or open_todos[:1] or ["Continue from the recent raw messages below."])[:3])
    L.append("preserved_trace:")
    s = max(0, len(prefix) - PRESERVED_TAIL_MSGS)
    header = "\n".join(L) + "\n"
    budget_chars = max(800, (budget.wc_cap - budget.text_tokens(header)) * 3)
    per = max(200, budget_chars // max(1, len(prefix) - s))
    trace = [f'<trace source_start="{s}" source_end="{len(prefix)}" messages="{len(prefix) - s}">']
    for m in prefix[s:]:
        body = message_text(m)
        if m["role"] == "assistant" and m.get("tool_calls"):
            body += " " + " ".join(f'[{tc["function"]["name"]} {safe_clip(tc["function"]["arguments"], 120)}]'
                                   for tc in m["tool_calls"])
        trace.append(f'[{m["role"]}] {safe_clip(body.strip(), per)}')
    trace.append("</trace>")
    L.append("\n".join(trace))
    L.append("</working_context>")
    return WC_PREAMBLE + "\n\n" + "\n".join(L)


# ----------------------------------------------------------------- tools
def infer_tool_registry(sessions):
    """Minimal OpenAI-style definitions from observed calls: property types and the
    keys present in every call. Replace with your harness's real definitions when you can."""
    props, required, calls = {}, {}, Counter()
    for _, msgs in sessions:
        for m in msgs:
            for tc in m.get("tool_calls") or []:
                name, a = tc["function"]["name"], _args(tc)
                calls[name] += 1
                p = props.setdefault(name, {})
                for k, v in a.items():
                    t = {bool: "boolean", int: "integer", float: "number", str: "string", list: "array", dict: "object"}.get(type(v), "string")
                    p.setdefault(k, set()).add(t)
                required[name] = set(a) if name not in required else required[name] & set(a)
    reg = {}
    for name in sorted(calls):
        schema = {"type": "object", "properties": {k: {"type": sorted(ts)[0] if len(ts) == 1 else sorted(ts)}
                                                   for k, ts in sorted(props[name].items())}}
        if required.get(name):
            schema["required"] = sorted(required[name])
        reg[name] = {"type": "function", "function": {"name": name, "description": f"{name} tool.", "parameters": schema}}
    return reg


def load_tool_registry(path):
    data = json.loads(Path(path).read_text())
    tools = data["tools"] if isinstance(data, dict) and "tools" in data else data
    reg = {}
    for t in tools:
        f = t["function"] if t.get("type") == "function" else t
        reg[f["name"]] = {"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                                             "parameters": f.get("parameters") or {"type": "object", "properties": {}}}}
    baseline = data.get("baseline", sorted(reg)) if isinstance(data, dict) else sorted(reg)
    return reg, baseline


def tools_for(messages, registry, baseline, inferred):
    called = {tc["function"]["name"] for m in messages for tc in m.get("tool_calls") or []}
    names = list(baseline) + sorted(called - set(baseline))
    out = []
    for n in names:
        if n in registry:
            out.append(registry[n])
        elif n in inferred:
            out.append(inferred[n])
        else:
            raise ValueError(f"no definition for tool {n}")
    return out


# -------------------------------------------------------------- assembly
class SessionBuilder:
    def __init__(self, budget, system_prompt, registry, baseline, inferred, declare_session_tools):
        self.b, self.system_prompt = budget, system_prompt
        self.registry, self.baseline, self.inferred = registry, baseline, inferred
        self.declare_session_tools = declare_session_tools

    def system_message(self, cwd, wc):
        text = ""
        if self.system_prompt:
            text = self.system_prompt + (ENV_BLOCK.format(cwd=cwd) if cwd else "")
        if wc:
            text = (text + "\n\n" + wc) if text else wc
        return {"role": "system", "content": text} if text else None

    def make_chunk(self, M, header, gen, a, b):
        window = M[a:b]
        msgs = []
        wc = build_working_context(M[:a], header["session_id"], gen, header.get("cwd"), self.b) if a > 0 else None
        sysm = self.system_message(header.get("cwd"), wc)
        if sysm:
            msgs.append(sysm)
        if a > 0 and window[0]["role"] != "user":
            msgs.append({"role": "user", "content": BRIDGE})
        msgs.extend(window)
        scope = M if self.declare_session_tools else window
        tools = tools_for(scope, self.registry, self.baseline, self.inferred)
        return {"chunk": gen, "src_range": [a, b], "continuation": a > 0, "messages": msgs, "tools": tools}

    def render_ranges(self, M, header, ranges):
        chunks = [self.make_chunk(M, header, g, a, b) for g, (a, b) in enumerate(ranges)]
        return chunks, [self.b.size(c["messages"], c["tools"]) for c in chunks]

    def rebalance_zero_loss(self, M, header, chunks, sizes):
        ranges = [tuple(c["src_range"]) for c in chunks]
        bounds = safe_boundaries(M)
        for _ in range(max(4, len(ranges) * 4)):
            chunks, sizes = self.render_ranges(M, header, ranges)
            zero = [i for i, (_, loss) in enumerate(sizes) if loss == 0]
            if not zero:
                return chunks, sizes
            i, cands = zero[0], []
            if i > 0:
                cands.append(ranges[:i - 1] + [(ranges[i - 1][0], ranges[i][1])] + ranges[i + 1:])
            if i + 1 < len(ranges):
                cands.append(ranges[:i] + [(ranges[i][0], ranges[i + 1][1])] + ranges[i + 2:])
            if i > 0:
                left, cur = ranges[i - 1], ranges[i]
                for s in reversed([x for x in bounds if left[0] < x < cur[0]]):
                    cands.append(ranges[:i - 1] + [(left[0], s), (s, cur[1])] + ranges[i + 1:])
            if i + 1 < len(ranges):
                cur, right = ranges[i], ranges[i + 1]
                for s in [x for x in bounds if cur[1] < x < right[1]]:
                    cands.append(ranges[:i] + [(cur[0], s), (s, right[1])] + ranges[i + 2:])
            accepted = None
            for cand in cands:
                if not exact_cover(cand, len(M)):
                    continue
                _, cs = self.render_ranges(M, header, cand)
                if all(n <= self.b.max_len for n, _ in cs) and sum(1 for _, l in cs if l == 0) < len(zero):
                    accepted = cand
                    break
            if accepted is None:
                raise ValueError(f"cannot rebalance zero-loss range {ranges[i]} in {header['session_id']}")
            ranges = accepted
        raise ValueError(f"zero-loss rebalance did not converge in {header['session_id']}")

    def build(self, header, M):
        tools_est = json.dumps(tools_for(M, self.registry, self.baseline, self.inferred), ensure_ascii=False)
        sysm = self.system_message(header.get("cwd"), None)
        overhead = (self.b.text_tokens(sysm["content"]) if sysm else 0) + self.b.text_tokens(tools_est) + SYS_MARGIN
        window = self.b.raw_window - overhead
        if window < self.b.floor:
            raise ValueError(f"system prompt + tools leave too little room ({window} tokens) in {header['session_id']}")
        pref = [0.0]                              # prefix sums of the per-message estimate
        for m in M:
            pref.append(pref[-1] + self.b.est_message(m))
        while True:
            chunks, sizes = self.render_ranges(M, header, pack(M, pref, window))
            if all(n <= self.b.max_len for n, _ in sizes):
                break
            if window - self.b.shrink < self.b.floor:
                over = [(c["src_range"], n) for c, (n, _) in zip(chunks, sizes) if n > self.b.max_len]
                raise ValueError(f"oversize window at floor in {header['session_id']}: {over}")
            window -= self.b.shrink
        chunks, sizes = self.rebalance_zero_loss(M, header, chunks, sizes)
        if not exact_cover([tuple(c["src_range"]) for c in chunks], len(M)):
            raise ValueError(f"non-exact coverage in {header['session_id']}")
        return chunks, sizes, window


def val_split(sizes, val_frac):
    """Whole sessions, seeded-hash order, filled up to val_frac of the corpus by size."""
    target = sum(sizes.values()) * val_frac
    order = sorted(sizes, key=lambda s: hashlib.sha256((SPLIT_SEED + s).encode()).hexdigest())
    val, acc = set(), 0
    for s in order:
        if acc + sizes[s] <= target:
            val.add(s)
            acc += sizes[s]
    if not val and len(sizes) > 1:
        val = {min(sizes, key=sizes.get)}
    return val


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--model", required=True, help="Tinker model name, e.g. Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--renderer", help="override the cookbook's recommended renderer name")
    ap.add_argument("--no-preserve-thinking", action="store_true", help="let hybrid renderers strip historical thinking")
    ap.add_argument("--effort", type=float, help="thinking effort for renderers that take one (Inkling); default 0.9")
    ap.add_argument("--max-len", type=int, default=65536, help="context length the rows must fit (match the model variant)")
    ap.add_argument("--system-prompt-file", type=Path, help="system prompt for every row; the working directory is appended")
    ap.add_argument("--tools-file", type=Path, help="OpenAI-style tool definitions (list, or {tools: [...], baseline: [...]})")
    ap.add_argument("--declare-session-tools", action="store_true",
                    help="declare every tool the session used on each of its rows (default: tools called in the row + baseline)")
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--name", help="dataset name (default: <model>-<ctx>k)")
    ap.add_argument("--limit", type=int, help="first N sessions only (smoke test)")
    args = ap.parse_args()

    src, out = args.data_dir / "anonymized", args.data_dir / "datasets"
    files = sorted(src.glob("*.jsonl"))[: args.limit]
    if not files:
        sys.exit(f"no sessions under {src}; run anonymize_sessions.py first")
    name = args.name or f"{model_slug(args.model)}-{args.max_len // 1024}k"
    system_prompt = args.system_prompt_file.read_text().strip() if args.system_prompt_file else ""

    tokenizer, renderer, prov = make_renderer(args.model, args.renderer, not args.no_preserve_thinking)
    if not prov["has_extension_property"]:
        print("note: this renderer reports no extension property. Rows are still rendered as one sequence per "
              "conversation with loss on every assistant turn (the cookbook logs a warning). For Qwen3.5/3.6 this "
              "is a documented token-level nuance that only affects assistant turns without thinking.", file=sys.stderr)
    prov["effort"] = (0.9 if args.effort is None else args.effort) if prov["accepts_effort"] else None
    budget = Budget(tokenizer, renderer, prov, args.effort, args.max_len)

    sessions = [read_session(p) for p in files]
    inferred = infer_tool_registry(sessions)
    registry, baseline = load_tool_registry(args.tools_file) if args.tools_file else ({}, [])
    builder = SessionBuilder(budget, system_prompt, registry, baseline, inferred, args.declare_session_tools)

    sizes = {h["session_id"]: p.stat().st_size for p, (h, _) in zip(files, sessions)}
    val = val_split(sizes, args.val_frac)
    out.mkdir(parents=True, exist_ok=True)
    rows = {"train": [], "val": []}
    manifest, t0 = [], time.time()
    for i, (path, (header, M)) in enumerate(zip(files, sessions)):
        split = "val" if header["session_id"] in val else "train"
        chunks, chunk_sizes, window = builder.build(header, M)
        for c, (n, nl) in zip(chunks, chunk_sizes):
            if n > args.max_len or nl <= 0:
                raise ValueError(f"{header['session_id']} chunk {c['chunk']}: tokens={n} loss={nl}")
            rows[split].append({"messages": c["messages"], "tools": c["tools"]})
            manifest.append({"split": split, "session": header["session_id"], "chunk": c["chunk"],
                             "src_range": c["src_range"], "continuation": c["continuation"],
                             "tokens": n, "loss_tokens": nl, "tools": [t["function"]["name"] for t in c["tools"]]})
        print(f"[{i + 1:3d}/{len(files)}] {header['session_id'][:36]:36s} {split:5s} {len(chunks):3d} rows "
              f"window {window}  ({time.time() - t0:4.0f}s)", flush=True)

    paths = {s: out / f"{name}.{s}.jsonl" for s in ("train", "val")}
    for s in paths:
        write_jsonl(paths[s], rows[s])
    toks = [m["tokens"] for m in manifest]
    report = {"name": name, "model": args.model, "max_len": args.max_len, "rendering": prov,
              "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest() if system_prompt else None,
              "tools_source": str(args.tools_file) if args.tools_file else "inferred-from-usage",
              "declare_session_tools": args.declare_session_tools,
              "files_sha256": {s: sha256_file(p) for s, p in paths.items()},
              "sessions": len(files), "val_sessions": sorted(val),
              "rows": {s: len(rows[s]) for s in rows},
              "tokens": {"max": max(toks), "mean": sum(toks) // len(toks), "total": sum(toks)},
              "tokens_by_split": {s: sum(m["tokens"] for m in manifest if m["split"] == s) for s in rows},
              "loss_tokens_total": sum(m["loss_tokens"] for m in manifest),
              "continuation_rows": sum(1 for m in manifest if m["continuation"]),
              "tool_names": sorted({t for m in manifest for t in m["tools"]})}
    write_json(out / f"{name}.manifest.json", manifest)
    write_json(out / f"{name}.report.json", report)
    print(json.dumps({k: report[k] for k in ("name", "rows", "tokens", "tokens_by_split", "loss_tokens_total",
                                              "continuation_rows", "val_sessions")}, indent=1))
    print(f"ALL ROWS <= {args.max_len} TOKENS, LOSS > 0, SESSIONS COVERED EXACTLY ONCE")


if __name__ == "__main__":
    main()
