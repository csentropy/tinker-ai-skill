#!/usr/bin/env python3
"""Step 1: copy Claude Code session transcripts into a dataset location.

Claude Code writes one JSONL file per session under
~/.claude/projects/<project-slug>/<session-uuid>.jsonl (subagent transcripts live
in <session-uuid>/subagents/). By default Claude Code deletes transcripts older
than 30 days -- set "cleanupPeriodDays" in ~/.claude/settings.json to something
large BEFORE you start collecting, or you will lose sessions.

This step copies files verbatim (sha256 recorded) and writes
<data-dir>/originals/manifest.json. Nothing is modified here; cleaning happens in
normalize_sessions.py and anonymize_sessions.py.

Selection: main sessions only (no subagent files), at least --min-assistant
assistant records, optional --project substring filter on the slug, optional
--since date, optional --exclude-slug prefixes (for example a client's project).

Usage:
    python scripts/collect_sessions.py --data-dir ./data
    python scripts/collect_sessions.py --claude-dir ~/.claude/projects --project my-repo --since 2026-06-01
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import sha256_file, write_json  # noqa: E402


def scan(path):
    """Session id, cwd, assistant record count, models, time range -- cheap pass."""
    sid, cwd, models, first, last, assistant, bad = None, Counter(), Counter(), None, None, 0, 0
    with open(path, "rb") as f:
        for raw in f:
            try:
                r = json.loads(raw)
            except ValueError:
                bad += 1
                continue
            if not isinstance(r, dict):
                continue
            sid = sid or r.get("sessionId")
            if r.get("cwd"):
                cwd[r["cwd"]] += 1
            ts = r.get("timestamp")
            if ts:
                first = min(first or ts, ts)
                last = max(last or ts, ts)
            if r.get("type") == "assistant" and not r.get("isSidechain"):
                assistant += 1
                model = (r.get("message") or {}).get("model")
                if model and model != "<synthetic>":
                    models[model] += 1
    return {"session_id": sid or path.stem, "cwd": cwd.most_common(1)[0][0] if cwd else None,
            "assistant_records": assistant, "models": dict(models), "first_timestamp": first,
            "last_timestamp": last, "unparseable_lines": bad}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--claude-dir", type=Path, default=Path.home() / ".claude" / "projects")
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--min-assistant", type=int, default=5, help="skip sessions with fewer assistant records")
    ap.add_argument("--min-bytes", type=int, default=20_000)
    ap.add_argument("--project", action="append", default=[], help="keep only slugs containing this substring (repeatable)")
    ap.add_argument("--exclude-slug", action="append", default=[], help="skip slugs starting with this prefix (repeatable)")
    ap.add_argument("--since", help="ISO date; skip sessions whose last record is older")
    args = ap.parse_args()

    if not args.claude_dir.is_dir():
        sys.exit(f"no such directory: {args.claude_dir}")
    out = args.data_dir / "originals"
    out.mkdir(parents=True, exist_ok=True)
    selected, skipped = [], Counter()
    # <slug>/<session>.jsonl only: subagent transcripts live one level deeper
    # (<slug>/<session>/subagents/*.jsonl) and are never matched by this glob.
    for path in sorted(args.claude_dir.glob("*/*.jsonl")):
        slug = path.parent.name
        if args.project and not any(p in slug for p in args.project):
            skipped["project_filter"] += 1
            continue
        if any(slug.startswith(x) for x in args.exclude_slug):
            skipped["excluded_slug"] += 1
            continue
        if path.stat().st_size < args.min_bytes:
            skipped["too_small_bytes"] += 1
            continue
        info = scan(path)
        if info["assistant_records"] < args.min_assistant:
            skipped["too_few_assistant_records"] += 1
            continue
        if args.since and (info["last_timestamp"] or "") < args.since:
            skipped["older_than_since"] += 1
            continue
        target = out / slug / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        size = target.stat().st_size
        selected.append({**info, "slug": slug, "file": str(target.relative_to(out)),
                         "bytes": size, "sha256": sha256_file(target)})
        print(f"copied {slug}/{path.name}  assistant={info['assistant_records']}  {size:,} bytes", flush=True)

    write_json(out / "manifest.json", {"source": str(args.claude_dir), "sessions": selected,
                                       "skipped": dict(skipped), "min_assistant": args.min_assistant})
    print(json.dumps({"selected": len(selected), "skipped": dict(skipped),
                      "bytes": sum(s["bytes"] for s in selected)}, indent=1))
    if not selected:
        sys.exit("no sessions selected")


if __name__ == "__main__":
    main()
