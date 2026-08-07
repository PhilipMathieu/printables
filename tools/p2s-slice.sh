#!/bin/bash
# p2s-slice -- slice a Bambu project 3mf for the P2S.
#
# Bambu Studio has to run in the user's GUI session, which a sandboxed command
# cannot reach. This wrapper is the one thing allowed to run outside it.
#
# INSTALL THIS TO ~/bin, NOT FROM THE REPO. Claude can write to the repo, so a
# repo copy could be rewritten into something this script is not; ~/bin is
# outside its sandbox write policy, so the copy there is the actual boundary.
#
#   mkdir -p ~/bin
#   cp tools/p2s-slice.sh ~/bin/p2s-slice
#   chmod 755 ~/bin/p2s-slice
#
# Then allow only that path in ~/.claude/settings.json:
#
#   "sandbox": { "excludedCommands": ["/Users/philip/bin/p2s-slice"] }
#
# The exclusion is deliberately a fixed script rather than the Bambu binary:
# this takes exactly one argument, runs one hard-coded command line, and
# refuses any path outside the project's out/ directory.

set -euo pipefail

BAMBU_APP="/Applications/BambuStudio.app"
PROJECT_ROOT="/Users/philip/src/printables"
ALLOWED_DIR="$PROJECT_ROOT/out"

die() { printf 'p2s-slice: %s\n' "$1" >&2; exit "${2:-1}"; }

[ $# -eq 1 ] || die "usage: p2s-slice <project.3mf>" 2
[ -d "$BAMBU_APP" ] || die "Bambu Studio not found at $BAMBU_APP" 3

src=$1
[ -f "$src" ] || die "no such file: $src" 4

# Resolve symlinks and '..' before checking, so the confinement can't be walked
# out of with a crafted relative path.
dir=$(cd -P -- "$(dirname -- "$src")" && pwd -P) || die "cannot resolve $src" 4
abs="$dir/$(basename -- "$src")"

case "$abs" in
  "$ALLOWED_DIR"/*.3mf) ;;
  *) die "refusing $abs -- only .3mf files under $ALLOWED_DIR may be sliced" 5 ;;
esac

outdir="$ALLOWED_DIR/sliced"
mkdir -p "$outdir"

# Launched straight from a terminal, Bambu Studio never joins the user's GUI
# session: it fails com.apple.hiservices-xpcservice lookups and stalls before
# slicing anything. Going through LaunchServices with `open` starts it in the
# Aqua session, where it slices in seconds. -W waits for exit, -n forces a
# fresh instance so an already-open window does not swallow the arguments.
# --arrange 0 is not optional: a nested print-in-place assembly is many
# disjoint solids, and the CLI otherwise spreads them across the bed as
# separate objects, destroying the nesting.
exec open -W -n -a "$BAMBU_APP" --args \
  --arrange 0 \
  --orient 0 \
  --slice 0 \
  --outputdir "$outdir" \
  "$abs"
