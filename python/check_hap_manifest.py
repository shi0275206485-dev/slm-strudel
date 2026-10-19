"""
check_hap_manifest.py

Quick diagnostic for a manifest produced by epoch_hap_check.mjs -- prints
the status breakdown and a sample of actual error messages, so you don't
have to guess why HAP_QUERY_SUCCESS came in lower than expected.

Usage:
    python check_hap_manifest.py <manifest.jsonl>
"""

import sys
import json
from collections import Counter

if len(sys.argv) < 2:
    print("Usage: python check_hap_manifest.py <manifest.jsonl>")
    sys.exit(1)

path = sys.argv[1]

with open(path, encoding='utf-8') as f:
    rows = [json.loads(l) for l in f if l.strip()]

statuses = Counter(r['status'] for r in rows)
print(f"Total entries: {len(rows)}")
print("Status breakdown:")
for status, count in statuses.most_common():
    print(f"  {status}: {count}")

print()
print("--- sample HAP_QUERY_ERROR messages (up to 10) ---")
errors = [r for r in rows if r['status'] == 'HAP_QUERY_ERROR']
for e in errors[:10]:
    print(f"  entry {e.get('entry_index')}: {e.get('message', '')}")

print()
print("--- sample STILL_NO_NOTE / other non-success entries (up to 5) ---")
others = [r for r in rows if r['status'] not in ('SUCCESS', 'HAP_QUERY_ERROR')]
for o in others[:5]:
    print(f"  entry {o.get('entry_index')}: status={o['status']} message={o.get('message', '')}")
