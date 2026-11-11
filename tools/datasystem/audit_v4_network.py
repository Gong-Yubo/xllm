#!/usr/bin/env python3
"""Gate a DataSystem re-test on fast-transport health before comparing stores.

The v4 bundle looked like a "store is slow" result, but the worker logs show
a systematic RDMA failure: nearly every cross-node read fell back to TCP, and
the few reads that did return a payload took ~64 s. Any E1 / performance
comparison done on that link is contaminated, so this tool turns the findings
into an explicit pre-flight gate.

It streams the (very large) worker and access logs and reports, per node:
  * REMOTE_MGET count and >=60 s share            (access.log)
  * UcpWrite ... fallback to tcp count           (worker INFO)
  * Get/RemotePull payload hit rate and p99 cost (worker INFO slow logs)
  * ucp_ep_create failures and RDMA timeouts     (worker INFO)

Gates (REANALYSIS_v4.md section 4, P0-1):
  fallback / REMOTE_MGET < 1%, hit p99 < 1000 ms, ucp_ep_create failures = 0.

Usage:
  python3 tools/datasystem/audit_v4_network.py
  python3 tools/datasystem/audit_v4_network.py --root /path/to/bundle --no-fail
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

BUNDLE = Path("logs/xllm_test_v4/bundle_v4_sanitized")
PAYLOAD_RE = re.compile(r"payload size: (\d+)")
COST_RE = re.compile(r"cost: ([\d.]+)ms")
KEY_HEX = "786c6c6d2d6b762d7633"  # hex("xllm-kv-v3")


def p99(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))]


def node_name(path: Path) -> str:
    # .../node2/datasystem/<file>
    return path.parent.parent.name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / BUNDLE)
    parser.add_argument("--fallback-budget", type=float, default=0.01,
                        help="max fallback/REMOTE_MGET ratio (default 0.01)")
    parser.add_argument("--p99-budget-ms", type=float, default=1000.0)
    parser.add_argument("--no-fail", action="store_true",
                        help="always exit 0, useful when inspecting an old bundle")
    args = parser.parse_args()
    root: Path = args.root.resolve()

    access = sorted(root.glob("node*/datasystem/access.log"))
    workers = sorted(root.glob("node*/datasystem/datasystem_worker.INFO.log"))
    if not access and not workers:
        print(f"RESULT: FAIL (no DataSystem logs under {root})")
        return 0 if args.no_fail else 1

    stats: dict[str, dict[str, object]] = defaultdict(lambda: {
        "remote_mget": 0, "remote_slow": 0, "fallback": 0,
        "hit_costs": [], "empty_costs": [], "ucp_errors": 0, "rdma_timeouts": 0,
    })

    for path in access:
        node = node_name(path)
        with path.open("rb") as handle:
            for raw in handle:
                if b"DS_POSIX_REMOTE_MGET" not in raw:
                    continue
                parts = raw.decode("utf-8", errors="replace").split(" | ")
                if len(parts) < 12 or KEY_HEX not in parts[11]:
                    continue
                # [7]=code (always 0/K_OK), [9]=microseconds, [10]=dataSize
                if parts[10].strip() != "4194304":  # 4 MiB KV block
                    continue
                stats[node]["remote_mget"] += 1  # type: ignore[operator]
                try:
                    usec = int(parts[9].strip())
                except ValueError:
                    continue
                if usec >= 60_000_000:
                    stats[node]["remote_slow"] += 1  # type: ignore[operator]

    for path in workers:
        node = node_name(path)
        with path.open("rb") as handle:
            for raw in handle:
                if b"fallback to tcp" in raw:
                    stats[node]["fallback"] += 1  # type: ignore[operator]
                if b"ucp_ep_create failed" in raw:
                    stats[node]["ucp_errors"] += 1  # type: ignore[operator]
                if b"Timed out waiting for request" in raw:
                    stats[node]["rdma_timeouts"] += 1  # type: ignore[operator]
                if b"[Get/RemotePull] finish" in raw:
                    line = raw.decode("utf-8", errors="replace")
                    payload = PAYLOAD_RE.search(line)
                    cost = COST_RE.search(line)
                    if not payload or not cost:
                        continue
                    bucket = ("hit_costs" if int(payload.group(1)) > 0
                              else "empty_costs")
                    stats[node][bucket].append(float(cost.group(1)))  # type: ignore[union-attr]

    header = (f"{'node':<10} {'remote_mget':>12} {'>=60s':>7} {'fallback':>10} "
              f"{'ratio':>8} {'pull_hits':>10} {'hit_p99_ms':>11} "
              f"{'ucp_err':>8} {'rdma_to':>8}")
    print(header)
    print("-" * len(header))
    totals = {"remote_mget": 0, "remote_slow": 0, "fallback": 0,
              "ucp_errors": 0, "rdma_timeouts": 0}
    all_hits: list[float] = []
    for node in sorted(stats):
        item = stats[node]
        remote = int(item["remote_mget"])
        fallback = int(item["fallback"])
        hits = list(item["hit_costs"])  # type: ignore[arg-type]
        ratio = fallback / remote if remote else float("inf") if fallback else 0.0
        print(f"{node:<10} {remote:>12} {int(item['remote_slow']):>7} "
              f"{fallback:>10} {ratio:>7.1%} {len(hits):>10} "
              f"{p99(hits):>11.1f} {int(item['ucp_errors']):>8} "
              f"{int(item['rdma_timeouts']):>8}")
        for key in totals:
            totals[key] += int(item[key])  # type: ignore[index]
        all_hits += hits

    remote = totals["remote_mget"]
    fallback = totals["fallback"]
    ratio = fallback / remote if remote else 0.0
    hit_p99 = p99(all_hits)
    print("-" * len(header))
    print(f"{'TOTAL':<10} {remote:>12} {totals['remote_slow']:>7} "
          f"{fallback:>10} {ratio:>7.1%} {len(all_hits):>10} "
          f"{hit_p99:>11.1f} {totals['ucp_errors']:>8} "
          f"{totals['rdma_timeouts']:>8}")

    failures = []
    if not remote:
        failures.append("no REMOTE_MGET 4 MiB records: link/store not exercised")
    if ratio >= args.fallback_budget:
        failures.append(f"TCP fallback ratio {ratio:.1%} >= "
                        f"{args.fallback_budget:.1%}")
    if all_hits and hit_p99 >= args.p99_budget_ms:
        failures.append(f"remote-pull hit p99 {hit_p99:.1f} ms >= "
                        f"{args.p99_budget_ms:.0f} ms")
    if not all_hits:
        failures.append("no remote pull returned a payload")
    if totals["ucp_errors"]:
        failures.append(f"ucp_ep_create failures = {totals['ucp_errors']}")

    print(f"GATES: fallback<{args.fallback_budget:.1%}, "
          f"hit_p99<{args.p99_budget_ms:.0f} ms, ucp_errors=0")
    for failure in failures:
        print(f"  FAIL: {failure}")
    print(f"RESULT: {'FAIL' if failures else 'PASS'} "
          f"({'do not use this link for a store comparison' if failures else 'link healthy'})")
    if failures and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
