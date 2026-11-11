#!/usr/bin/env python3
"""Audit YuanRong request-timeout values before a deployment / re-test.

The v3 launch scripts passed a 300000 ms request timeout explicitly.
That bypassed the new 500 ms default and would restore the minute-long stalls
seen in v4. The C++ side now clamps both ends (see
store_config_validation.h), but a stale command line or JSON config is still
worth catching before a 32-card run.

Checks:
  1. The six in-tree default declarations all equal
     kDefaultStoreRequestTimeoutMs and stay within kMaxStoreRequestTimeoutMs.
  2. Explicit values in deployment files/arguments are inside [1, max].
  3. An explicit --value N (optional) is inside [1, max].

Usage:
  python3 tools/datasystem/audit_store_timeout.py
  python3 tools/datasystem/audit_store_timeout.py --value 300000
  python3 tools/datasystem/audit_store_timeout.py --path deploy/start.sh
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

POLICY_HEADER = Path("xllm/core/kv_cache/transfer/store_config_validation.h")
SCAN_SUFFIXES = {".sh", ".bash", ".json", ".yaml", ".yml", ".toml", ".conf", ".py"}
SKIP_DIRS = {".git", "third_party", "output", "logs", "build", "node_modules", "__pycache__"}

DEFAULT_RE = re.compile(r"kDefaultStoreRequestTimeoutMs\s*=\s*(\d+)")
MAX_RE = re.compile(r"kMaxStoreRequestTimeoutMs\s*=\s*(\d+)")
PROPERTY_RE = re.compile(
    r"PROPERTY\(int32_t,\s*store_yuanrong_timeout_ms\)\s*=\s*(-?\d+)")
CONFIG_RE = re.compile(r"int32_t\s+yuanrong_timeout_ms\s*=\s*(-?\d+)")
DEFINE_RE = re.compile(
    r"DEFINE_int32\(\s*store_yuanrong_timeout_ms\s*,\s*(-?\d+)", re.S)
EXPLICIT_RE = re.compile(
    r"store_yuanrong_timeout_ms\s*(?:\)|\"|'|\s)*[=:]\s*\"?'?(-?\d+)")


def parse_policy(root: Path) -> tuple[int, int]:
    text = (root / POLICY_HEADER).read_text(encoding="utf-8")
    default = DEFAULT_RE.search(text)
    maximum = MAX_RE.search(text)
    if not default or not maximum:
        raise SystemExit(f"cannot parse timeout policy from {root / POLICY_HEADER}")
    return int(default.group(1)), int(maximum.group(1))


def audit_source_defaults(root: Path, default: int, maximum: int) -> list[str]:
    """Every in-tree declaration must stay on the shared constant."""
    problems: list[str] = []
    found = 0
    for path in sorted((root / "xllm").rglob("*.h")) + sorted(
            (root / "xllm" / "core" / "framework" / "config").rglob("*.cpp")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in (PROPERTY_RE, CONFIG_RE, DEFINE_RE):
            for match in pattern.finditer(text):
                found += 1
                value = int(match.group(1))
                if value != default or not (0 < value <= maximum):
                    line = text[:match.start()].count("\n") + 1
                    problems.append(
                        f"{path.relative_to(root)}:{line} default={value}, "
                        f"expected {default} within [1, {maximum}]")
    if found < 6:
        problems.append(
            f"only {found} in-tree default declarations found (expected >= 6); "
            "the audit patterns may have drifted")
    return problems


def audit_paths(paths: list[Path], maximum: int, root: Path) -> tuple[list[str], int]:
    problems: list[str] = []
    checked = 0
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded += [p for p in path.rglob("*")
                         if p.is_file() and p.suffix in SCAN_SUFFIXES
                         and not SKIP_DIRS.intersection(p.parts)]
        else:
            expanded.append(path)
    for path in expanded:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            problems.append(f"cannot read {path}: {error}")
            continue
        checked += 1
        for number, line in enumerate(text.splitlines(), 1):
            for match in EXPLICIT_RE.finditer(line):
                value = int(match.group(1))
                if value <= 0 or value > maximum:
                    problems.append(
                        f"{path.relative_to(root) if path.is_relative_to(root) else path}"
                        f":{number} explicit store_yuanrong_timeout_ms={value} "
                        f"outside [1, {maximum}] -> {line.strip()[:160]}")
    return problems, checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(__file__).resolve().parents[2],
                        help="repository root (default: inferred from this script)")
    parser.add_argument("--path", type=Path, action="append", default=[],
                        help="extra file or directory to audit (repeatable)")
    parser.add_argument("--value", type=int, default=None,
                        help="check one explicit flag value")
    parser.add_argument("--no-scan", action="store_true",
                        help="skip the default repository-wide file scan")
    args = parser.parse_args()

    root = args.root.resolve()
    default, maximum = parse_policy(root)
    print(f"policy: default={default} ms, clamp=[1, {maximum}] ms "
          f"({POLICY_HEADER})")

    problems = audit_source_defaults(root, default, maximum)
    print(f"in-tree defaults: {'OK' if not problems else 'DRIFT'} "
          f"({len(problems)} finding(s))")

    paths: list[Path] = []
    if not args.no_scan:
        paths.append(root)
    paths += [p.resolve() for p in args.path]
    if paths:
        scan_problems, checked = audit_paths(paths, maximum, root)
        problems += scan_problems
        print(f"deployment files: scanned={checked}, "
              f"findings={len(scan_problems)}")

    if args.value is not None:
        ok = 0 < args.value <= maximum
        print(f"explicit value: {args.value} ms -> "
              f"{'OK' if ok else f'would be clamped to {maximum}; fix the flag'}")
        if not ok:
            problems.append(f"--value {args.value} outside [1, {maximum}]")

    for problem in problems:
        print(f"  FINDING: {problem}")
    print(f"RESULT: {'PASS' if not problems else 'FAIL'} "
          f"({len(problems)} finding(s))")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
