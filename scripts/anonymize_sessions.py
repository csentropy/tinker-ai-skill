#!/usr/bin/env python3
"""Step 2b: remove credentials and de-identify people and paths.

Input:  <data-dir>/normalized/*.jsonl      Output: <data-dir>/anonymized/*.jsonl + manifest.json

Two passes, both by value, never by executing anything in the transcript:

1. Credentials, corpus-wide. Provider API keys (sk-..., ghp_..., hf_..., AIza...,
   tml-...), JWTs, Bearer tokens, passwords in connection URLs, private-key blocks,
   and high-entropy values assigned to names like *_KEY / *_TOKEN / password are
   collected across ALL sessions first, then every occurrence is replaced by
   REDACTED_CREDENTIAL_<sha256 prefix>. One value maps to one marker everywhere, so a
   key printed by `cat .env` and later pasted into a command gets the same marker.
   Values are never printed or written anywhere except as hashes.

2. Identity, per session. Home-directory usernames are detected from paths
   (/Users/<u>/..., /home/<u>/..., project slugs -Users-<u>-...) and combined with
   the names, emails, handles, organisations and domains you list in an identity
   file. Each session draws its own fake identity (seeded by the session id, so a
   rebuild is reproducible) and its own names for the top-level folders under the
   home directory. The mapping is coherent inside a session and different between
   sessions ON PURPOSE: a constant fake world (/home/user/agents/...) in every row
   is a memorizable feature, and a model trained on it later asserted that world
   over the real runtime environment. Repository names and generic folders
   (src, scripts, Desktop, ...) keep their names.

After rewriting, every session is scanned for any remaining identity string or
credential candidate; the run fails (exit 1) if anything is left.

Identity file (JSON, all keys optional):
  {"usernames": ["jdoe"], "names": ["Jane Doe"], "emails": ["jane@example.com"],
   "handles": ["janedoe", "jdoe-dev"], "orgs": ["acme-robotics"], "domains": ["acme.com"]}

Usage:
    python scripts/anonymize_sessions.py --data-dir ./data --identity-file identity.json
    python scripts/anonymize_sessions.py --data-dir ./data --username jdoe --name "Jane Doe"
"""
import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump, read_session, sha256_file, sha256_text, write_json, write_jsonl  # noqa: E402

# ------------------------------------------------------------- credentials
FORMATS = {
    "provider_key": re.compile(r"(?<![\w/-])(?:sk-(?:proj-|svcacct-|ant-api\d\d-)?[A-Za-z0-9_-]{32,}|xai-[A-Za-z0-9_-]{40,}"
                               r"|tml-[A-Za-z0-9_-]{40,}|AIza[\w-]{35}|hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}"
                               r"|github_pat_[A-Za-z0-9_]{50,}|AKIA[0-9A-Z]{16}|xox[baprs]-[A-Za-z0-9-]{20,})(?![\w/-])"),
    "jwt": re.compile(r"(?<![\w-])eyJ[\w-]{12,}\.[\w-]{12,}\.[\w-]{12,}(?![\w-])"),
    "bearer": re.compile(r"\bBearer\s+([A-Za-z0-9._~+/=-]{20,})", re.I),
    "url_password": re.compile(r"\b(?:postgres(?:ql)?|mysql|redis|mongodb(?:\+srv)?|amqp|https?)://[^\s:@/]+:([^\s@/]{8,})@"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]{80,}?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
}
ASSIGN = re.compile(r"""\b([A-Za-z][A-Za-z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD)|api_key|apikey|token|password|client_secret"""
                    r"""|accessToken|refreshToken|id_token)\b["']?\s*[:=]\s*(["']?)([A-Za-z0-9][A-Za-z0-9_./+=:-]{7,})""", re.I)
# Obvious placeholders are not credentials.
FIXTURE = re.compile(r"example|placeholder|redacted|dummy|your[_-]|change.?me|xxxx|fixture|synthetic|fake|replace.with|<[^>]+>", re.I)
REFERENCE = re.compile(r"^(?:self|args|config|settings|process|os|env|secrets|payload|request|response|options)[._A-Z]", re.I)


def entropy(value):
    return -sum((n / len(value)) * math.log2(n / len(value)) for n in Counter(value).values())


def credential_candidates(text):
    seen = set()
    for _ in range(2):                       # look through one level of quote escaping
        text = text.replace('\\"', '"').replace("\\'", "'")
    for name, pattern in FORMATS.items():
        for m in pattern.finditer(text):
            value = m[1] if name in {"bearer", "url_password"} else m[0]
            if FIXTURE.search(value):
                continue
            if name == "bearer" and (REFERENCE.search(value) or entropy(value) < 3.5):
                continue
            if name == "url_password" and (len(value) < 16 or entropy(value) < 3.5):
                continue
            if value not in seen:
                seen.add(value)
                yield value, name
    for m in ASSIGN.finditer(text):
        _, quote, value = m.groups()
        if FIXTURE.search(value) or REFERENCE.search(value):
            continue
        if value.startswith(("/", "http", "secret://", "$")) or not re.search(r"\d", value):
            continue                        # paths, URLs, variable references, plain words
        if not quote and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value) and not re.search(r"[A-Z].*[a-z].*\d|\d.*[a-z].*[A-Z]", value):
            continue                        # an identifier, not a literal
        if len(value) < 16 or entropy(value) < 3.5:
            continue
        if value not in seen:
            seen.add(value)
            yield value, "assignment"


# ---------------------------------------------------------------- identity
GIVEN = ["alex", "jing", "daniel", "priya", "marco", "thanh", "kara", "ravi", "edwin", "yosuke", "felix", "colette",
         "sanjay", "nora", "lena", "wesley", "amara", "ezra", "nina", "jonas", "riya", "tomas", "greta", "idris",
         "mateo", "hana", "oliver", "sofia", "kenji", "leila", "aaron", "maya", "dmitri", "ines", "kofi", "elena"]
SURNAMES = ["kemp", "chen", "vorak", "raman", "solis", "nguyen", "whitfield", "krishnan", "elmore", "tanaka", "braun",
            "marchand", "gupta", "obrien", "hale", "fong", "diallo", "feld", "cole", "hoffmann", "patel", "mercado",
            "klein", "bello", "okafor", "lindqvist", "moreau", "sato", "novak", "haddad", "castro", "eriksen"]
DOMAINS = ["fastmail.com", "proton.me", "gmail.com", "hey.com", "icloud.com", "outlook.com"]
ORG_A = ["kestrel", "bluepine", "northfield", "quarry", "veldt", "lantern", "halcyon", "ridgeline", "copperleaf",
         "arclight", "stonebrook", "tidewater", "foxglove", "meridian", "saltmarsh"]
ORG_B = ["labs", "systems", "ai", "works", "dev", "io", "cloud", "tech", "forge", "hq"]
CONTAINERS = ["engineering", "lab", "sandbox", "stacks", "ws", "mono", "forge", "workshop", "hack", "apps", "services",
              "platform", "research", "experiments", "tools", "oss", "clients", "infra", "ml", "scratch", "products",
              "studio", "bench", "garage", "notebooks", "prototypes", "builds", "codebase", "sources", "personal"]
STANDARD_DIRS = {"Desktop", "Downloads", "Documents", "Library", "Movies", "Music", "Pictures", "Public", "Applications"}
GENERIC_DIRS = {"bin", "src", "scripts", "data", "tmp", "go", "node_modules", "models", "packages", "work", "code",
                "dev", "projects", "repos", "workspace", "git", "opt", "var", "etc", "local", "snap"}
SYSTEM_USERS = {"Shared", "Guest", "Default", "root", "runner", "ubuntu", "ec2-user", "user", "admin", "vagrant", "node"}
HOME_RE = re.compile(r"/(Users|home)/([A-Za-z][A-Za-z0-9._-]{1,31})(?![A-Za-z0-9._-])")
SLUG_RE = re.compile(r"-(Users|home)-([A-Za-z][A-Za-z0-9._]{1,31})(?=-|(?![A-Za-z0-9._]))")
SEG_END = r"(?![A-Za-z0-9_.-]*[A-Za-z0-9_])"


def fake_identity(rng):
    given, surname = rng.choice(GIVEN), rng.choice(SURNAMES)
    username = rng.choice([given[0] + surname, given, given + surname[0], surname + given[0], given + surname,
                           given[:3] + surname[:3]])
    handle = rng.choice([given + surname, given[0] + surname, surname + given[0], given + "dev"])
    org = rng.choice(ORG_A) + "-" + rng.choice(ORG_B)
    local = rng.choice([given + "." + surname, given[0] + surname, given])
    return {"given": given, "surname": surname, "username": username, "handle": handle, "org": org,
            "fullname": f"{given.capitalize()} {surname.capitalize()}",
            "email": f"{local}@{rng.choice(DOMAINS)}", "work_domain": org.replace("-", "") + ".com"}


def detect_usernames(texts):
    found = Counter()
    for t in texts:
        for _, u in HOME_RE.findall(t):
            found[u] += 1
        for _, u in SLUG_RE.findall(t):
            found[u] += 1
    return {u: n for u, n in found.items() if u not in SYSTEM_USERS}


def container_mapping(text, usernames, rng):
    """Top-level folders under the home directory that carry identity (not standard,
    not generic) -> random container names, collision-free with real folder names."""
    homes = "|".join([re.escape(f"/Users/{u}") for u in usernames] + [re.escape(f"/home/{u}") for u in usernames]
                     + [r"~", r"\$HOME"])
    dir_re = re.compile(rf"(?:{homes})/([A-Za-z0-9_][A-Za-z0-9_.-]*?)/")
    segs = {s for s in dir_re.findall(text) if not s.startswith(".") and s not in STANDARD_DIRS and s not in GENERIC_DIRS}
    taken = set(dir_re.findall(text))
    pool = [c for c in rng.sample(CONTAINERS, k=len(CONTAINERS)) if c not in taken]
    return {seg: pool[i] if i < len(pool) else f"{pool[i % len(pool)]}{i}" for i, seg in enumerate(sorted(segs))}


def case_variants(old, new):
    """lower / Capitalized / UPPER spellings of one literal."""
    out = [(old, new)]
    if old.lower() != old:
        out.append((old.lower(), new.lower()))
    out.append((old.capitalize(), new.capitalize()))
    out.append((old.upper(), new.upper()))
    seen, uniq = set(), []
    for o, n in out:
        if o not in seen:
            seen.add(o)
            uniq.append((o, n))
    return uniq


def build_rules(ident, usernames, mapping, identity):
    """Ordered (regex, replacement) rules for one session. Most specific first."""
    u = ident["username"]
    rules = []
    if mapping:
        alt = "|".join(re.escape(s) for s in sorted(mapping, key=len, reverse=True))
        homes = "|".join([re.escape(f"/Users/{x}") for x in usernames] + [re.escape(f"/home/{x}") for x in usernames]
                         + [r"~", r"\$HOME"])

        def home_fix(prefix):
            m = HOME_RE.fullmatch(prefix)
            return f"/{m[1]}/{u}" if m else prefix
        rules.append((re.compile(rf"({homes})/({alt}){SEG_END}"), lambda m: f"{home_fix(m[1])}/{mapping[m[2]]}"))
        for real in usernames:
            rules.append((re.compile(rf"-(Users|home)-{re.escape(real)}-({alt})(?=-|[^A-Za-z0-9_.]|$)"),
                          lambda m: f"-{m[1]}-{u}-{mapping[m[2]]}"))
    for real in sorted(usernames, key=len, reverse=True):
        rules.append((re.compile(rf"/(Users|home)/{re.escape(real)}(?![A-Za-z0-9._-])"), lambda m: f"/{m[1]}/{u}"))
        rules.append((re.compile(rf"-(Users|home)-{re.escape(real)}(?=-|(?![A-Za-z0-9._]))"), lambda m: f"-{m[1]}-{u}"))
    literal = []
    for email in identity.get("emails", []):
        literal.append((email, ident["email"]))
    for name in identity.get("names", []):
        literal.append((name, ident["fullname"]))
        tokens = [t for t in re.split(r"\s+", name) if len(t) >= 3]
        if len(tokens) >= 2:
            literal += case_variants(tokens[0], ident["given"]) + case_variants(tokens[-1], ident["surname"])
    for handle in identity.get("handles", []):
        literal += case_variants(handle, ident["handle"])
    for org in identity.get("orgs", []):
        literal += case_variants(org, ident["org"])
        literal += case_variants(org.replace("-", ""), ident["org"].replace("-", ""))
        literal += case_variants(org.replace("-", " ").title(), ident["org"].replace("-", " ").title())
    for domain in identity.get("domains", []):
        literal.append((domain, ident["work_domain"]))
    for real in usernames:
        literal += case_variants(real, u)
    for old, new in sorted(literal, key=lambda p: len(p[0]), reverse=True):
        # alphanumeric boundaries: matches jdoe, jdoe_notes, ~jdoe, but not "sample" inside "samples"
        rules.append((re.compile(rf"(?<![A-Za-z0-9]){re.escape(old)}(?![A-Za-z0-9])"), new))
    return rules


def leak_pattern(usernames, identity, allow=()):
    """Case-insensitive scan for anything that still identifies the person. Opaque
    strings (usernames, emails, handles, orgs, domains) match anywhere; name tokens
    match as whole words so a surname that is also an English word does not flag
    every sentence. --allow-token removes a literal from the scan."""
    opaque = set(usernames)
    for key in ("emails", "handles", "orgs", "domains"):
        opaque.update(identity.get(key, []))
    words = set()
    for name in identity.get("names", []):
        opaque.add(name)
        words.update(t for t in re.split(r"\s+", name) if len(t) >= 4)
    allow = {a.lower() for a in allow}
    opaque = {x for x in opaque if len(x) >= 3 and x.lower() not in allow}
    words = {x for x in words if x.lower() not in allow and x.lower() not in {o.lower() for o in opaque}}
    parts = [re.escape(x) for x in sorted(opaque, key=len, reverse=True)]
    parts += [rf"(?<![A-Za-z0-9]){re.escape(x)}(?![A-Za-z0-9])" for x in sorted(words, key=len, reverse=True)]
    if not parts:
        return None
    return re.compile("|".join(parts), re.I)


# ----------------------------------------------------------------- rewriting
def texts_of(msg):
    """Yield (setter, text) for every string in a message that may hold content."""
    if msg["role"] in ("user", "tool") and isinstance(msg.get("content"), str):
        yield (lambda v, m=msg: m.__setitem__("content", v)), msg["content"]
    elif msg["role"] == "assistant":
        for part in msg.get("content") or []:
            for key in ("thinking", "text"):
                if isinstance(part.get(key), str):
                    yield (lambda v, p=part, k=key: p.__setitem__(k, v)), part[key]
        for tc in msg.get("tool_calls") or []:
            yield (lambda v, t=tc: t["function"].__setitem__("arguments", v)), tc["function"]["arguments"]


def rewrite(msgs, fn):
    n = 0
    for m in msgs:
        for setter, text in texts_of(m):
            new, k = fn(text)
            if k:
                setter(new)
                n += k
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("./data"))
    ap.add_argument("--identity-file", type=Path, help="JSON with usernames/names/emails/handles/orgs/domains")
    ap.add_argument("--username", action="append", default=[], help="extra real home-directory username (repeatable)")
    ap.add_argument("--name", action="append", default=[], help="extra real full name (repeatable)")
    ap.add_argument("--email", action="append", default=[], help="extra real email address (repeatable)")
    ap.add_argument("--handle", action="append", default=[], help="extra real handle, e.g. GitHub login (repeatable)")
    ap.add_argument("--org", action="append", default=[], help="extra real organisation name (repeatable)")
    ap.add_argument("--domain", action="append", default=[], help="extra real domain (repeatable)")
    ap.add_argument("--keep-username", action="append", default=[], help="detected username that is NOT a person (repeatable)")
    ap.add_argument("--salt", default="", help="changes every fake identity; use a new salt per public release")
    ap.add_argument("--allow-token", action="append", default=[], help="literal to exclude from the leak scan (repeatable)")
    args = ap.parse_args()

    src, out = args.data_dir / "normalized", args.data_dir / "anonymized"
    files = sorted(p for p in src.glob("*.jsonl"))
    if not files:
        sys.exit(f"no sessions under {src}; run normalize_sessions.py first")
    identity = json.loads(args.identity_file.read_text()) if args.identity_file else {}
    identity.setdefault("usernames", []).extend(args.username)
    identity.setdefault("names", []).extend(args.name)
    identity.setdefault("emails", []).extend(args.email)
    identity.setdefault("handles", []).extend(args.handle)
    identity.setdefault("orgs", []).extend(args.org)
    identity.setdefault("domains", []).extend(args.domain)

    sessions = {p: read_session(p) for p in files}
    all_texts = [t for _, (_, msgs) in sessions.items() for m in msgs for _, t in texts_of(m)]
    all_texts += [h.get("cwd") or "" for _, (h, _) in sessions.items()]

    # Pass 1: credentials by value, corpus-wide.
    values = {}
    for t in all_texts:
        for value, kind in credential_candidates(t):
            values.setdefault(value, set()).add(kind)
    markers = {v: f"REDACTED_CREDENTIAL_{sha256_text(v)[:12]}" for v in values}
    ordered = sorted(markers, key=len, reverse=True)

    def redact(text):
        n = 0
        for v in ordered:
            if v in text:
                n += text.count(v)
                text = text.replace(v, markers[v])
        return text, n

    # Pass 2: identity, per session.
    usernames = detect_usernames(all_texts)
    for u in args.keep_username:
        usernames.pop(u, None)
    usernames = {**{u: 0 for u in identity["usernames"]}, **usernames}
    leak_re = leak_pattern(usernames, identity, args.allow_token)
    print(f"credential values found: {len(values)}  ({dict(Counter(k for ks in values.values() for k in ks))})")
    print(f"home usernames detected: {sorted(usernames)}  (pass --keep-username X for any that is not a person)")

    out.mkdir(parents=True, exist_ok=True)
    report, leaks_total, cred_left_total = [], Counter(), 0
    for path, (header, msgs) in sessions.items():
        sid = header["session_id"]
        rng = random.Random(int(hashlib.sha256((args.salt + sid).encode()).hexdigest(), 16))
        ident = fake_identity(rng)
        joined = "\n".join(t for m in msgs for _, t in texts_of(m)) + "\n" + (header.get("cwd") or "")
        mapping = container_mapping(joined, list(usernames), rng) if usernames else {}
        rules = build_rules(ident, list(usernames), mapping, identity)

        def deidentify(text):
            n = 0
            for rx, new in rules:
                text, k = rx.subn(new, text)
                n += k
            return text, n

        n_cred = rewrite(msgs, redact)
        n_ident = rewrite(msgs, deidentify)
        if header.get("cwd"):
            header["cwd"] = deidentify(redact(header["cwd"])[0])[0]
        # Provenance must not leak either: keep only the transcript's file name
        # (a session uuid), never the project slug or the real home directory.
        if header.get("source_file"):
            header["source_file"] = Path(header["source_file"]).name
        after = dump(header) + "\n" + "\n".join(t for m in msgs for _, t in texts_of(m))
        leaks = Counter(m.group(0).lower() for m in leak_re.finditer(after)) if leak_re else Counter()
        cred_left = sum(1 for _ in credential_candidates(after))
        leaks_total.update(leaks)
        cred_left_total += cred_left
        header = {**header, "anonymization": {"username": ident["username"], "containers": mapping,
                                              "credential_replacements": n_cred, "identity_replacements": n_ident,
                                              "residual_leaks": dict(leaks), "residual_credentials": cred_left}}
        target = out / path.name
        write_jsonl(target, [header] + msgs)
        report.append({"session_id": sid, "file": target.name, "sha256": sha256_file(target),
                       "fake_username": ident["username"], "containers": mapping,
                       "credential_replacements": n_cred, "identity_replacements": n_ident,
                       "residual_leaks": dict(leaks), "residual_credentials": cred_left})
        print(f"{sid}: user={ident['username']} containers={mapping} credentials={n_cred} identity={n_ident}"
              f" leaks={dict(leaks)} residual_credentials={cred_left}", flush=True)

    write_json(out / "manifest.json", {
        "sessions": report, "salt_used": bool(args.salt),
        "real_usernames_detected": sorted(usernames),
        "credentials": {"unique_values": len(values), "kinds": dict(Counter(k for ks in values.values() for k in ks)),
                        "markers": sorted(markers.values())},
        "residual_leaks": dict(leaks_total), "residual_credentials": cred_left_total,
        "distinct_fake_usernames": len({r["fake_username"] for r in report})})
    print(dump({"sessions": len(report), "credential_values": len(values), "residual_leaks": dict(leaks_total),
                "residual_credentials": cred_left_total}))
    if leaks_total or cred_left_total:
        print("FAILED: identity strings or credential candidates remain; add them to the identity file or extend the patterns", file=sys.stderr)
        sys.exit(1)
    print("CLEAN: no identity strings or credential candidates remain")


if __name__ == "__main__":
    main()
