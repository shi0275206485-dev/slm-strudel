"""
patch_hap_bundle.py

Fixes multi-track hap extraction in hap_query_bundle.mjs.

The problem
-----------
Strudel's transpiler rewrites each `$:` line into a `.p(name)` call. The bundle
stubs that call as an identity function:

    f2.prototype.p = function(name) { return this; };

Every `$:` statement still evaluates, but only the LAST one survives as the
expression value, so `queryArc` sees a single track. Measured on the reference
set: entry 17 has a clap pattern and a triangle melody and returns 20 haps, all
triangle; entry 10 has four tracks and returns four haps, all clap. 21 of the
23 multi-track reference items show only one sound source.

The fix
-------
Make `.p` register each pattern, then query all of them and concatenate the
results. Concatenating per-track queries is equivalent to querying a stack for
this purpose and avoids having to locate the minified `stack` symbol.

Usage
-----
    python patch_hap_bundle.py hap_query_bundle.mjs
    python patch_hap_bundle.py hap_query_bundle.mjs --revert

Writes a .bak alongside the original on first run. Idempotent: running twice
does nothing the second time.
"""

import argparse
import os
import shutil
import sys

STUB_OLD = """f2.prototype.p = function(name) {
  return this;
};"""

STUB_NEW = """var __trackRegistry = [];
f2.prototype.p = function(name) {
  __trackRegistry.push(this);
  return this;
};"""

QUERY_OLD = """  const { pattern } = await On(code, F3);
  const haps = pattern.queryArc(0, numCycles);"""

QUERY_NEW = """  __trackRegistry = [];
  const { pattern } = await On(code, F3);
  const __targets = __trackRegistry.length > 1 ? __trackRegistry : [pattern];
  const haps = __targets.flatMap((t) => t.queryArc(0, numCycles));"""


def patch(path):
    src = open(path, "r", encoding="utf-8").read()

    if "__trackRegistry" in src:
        print("Already patched -- nothing to do.")
        return 0

    missing = []
    if STUB_OLD not in src:
        missing.append("the .p identity stub")
    if QUERY_OLD not in src:
        missing.append("the queryArc call in getHaps")
    if missing:
        print("Could not find " + " and ".join(missing) + ".")
        print("The bundle may have been rebuilt with different minified names.")
        print("Look near the end of the file for the 'hap_query_entry.mjs' section.")
        return 1

    backup = path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"Backup written to {backup}")

    src = src.replace(STUB_OLD, STUB_NEW, 1)
    src = src.replace(QUERY_OLD, QUERY_NEW, 1)
    open(path, "w", encoding="utf-8").write(src)

    print("Patched. `.p` now registers each track and all of them are queried.")
    print("\nCheck it with a two-track case, for example entry 17:")
    print("    $: s(`~ cp ~ cp`)")
    print("    $: n(`0 2 4 5 7`).scale(`D:dorian`).s(`triangle`)")
    print("  before: 20 haps, all triangle")
    print("  after : 28 haps, clap and triangle")
    return 0


def revert(path):
    backup = path + ".bak"
    if not os.path.exists(backup):
        print(f"No backup at {backup}.")
        return 1
    shutil.copy2(backup, path)
    print(f"Restored from {backup}.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", help="path to hap_query_bundle.mjs")
    ap.add_argument("--revert", action="store_true", help="restore the .bak copy")
    args = ap.parse_args()

    if not os.path.exists(args.bundle):
        print(f"No such file: {args.bundle}")
        return 1
    return revert(args.bundle) if args.revert else patch(args.bundle)


if __name__ == "__main__":
    sys.exit(main())
