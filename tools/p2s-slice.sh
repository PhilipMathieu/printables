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
# refuses any path outside the project.
#
# WHAT COUNTS AS INSIDE THE PROJECT. Work happens in git worktrees under
# .claude/worktrees/<name>/, each with its own out/, so confining to a single
# top-level out/ meant every worktree had to copy its 3mf up to the main
# checkout before it could be sliced -- across a boundary the sandbox exists to
# stop it crossing, which made the user the courier for exactly the file the
# rule was meant to scrutinise. The rule is now a property rather than a list
# of directories:
#
#   1. the path resolves to somewhere under PROJECT_ROOT,
#   2. its parent directory is named exactly "out",
#   3. it ends in .3mf.
#
# That readmits the main out/, admits every present and future worktree, and
# still cannot name a file outside the project. Resolution happens before the
# check -- both the root and the argument's directory go through `cd -P` --
# so symlinks and '..' are gone by the time anything is compared, and a
# basename can never contain a slash. Note the check is on where the file
# sits, not on who wrote it: anything able to write into the project can
# already put a 3mf in an out/ directory, so this bounds which file Bambu
# Studio is pointed at, not whether its contents are trusted.

set -euo pipefail

BAMBU_APP="/Applications/BambuStudio.app"
PROJECT_ROOT="/Users/philip/src/printables"

die() { printf 'p2s-slice: %s\n' "$1" >&2; exit "${2:-1}"; }

[ $# -eq 1 ] || die "usage: p2s-slice <project.3mf>" 2
[ -d "$BAMBU_APP" ] || die "Bambu Studio not found at $BAMBU_APP" 3

src=$1
[ -f "$src" ] || die "no such file: $src" 4

# Resolve the root too, not just the argument: if PROJECT_ROOT itself is
# reached through a symlink the two sides of the prefix test are written in
# different vocabularies and every path looks foreign.
root=$(cd -P -- "$PROJECT_ROOT" && pwd -P) || die "cannot resolve $PROJECT_ROOT" 3

# Resolve symlinks and '..' before checking, so the confinement can't be walked
# out of with a crafted relative path.
dir=$(cd -P -- "$(dirname -- "$src")" && pwd -P) || die "cannot resolve $src" 4
abs="$dir/$(basename -- "$src")"

case "$abs" in
  "$root"/*.3mf) ;;
  *) die "refusing $abs -- only .3mf files under $root may be sliced" 5 ;;
esac

[ "$(basename -- "$dir")" = "out" ] ||
  die "refusing $abs -- a project 3mf must sit in a directory named 'out'" 5

# Only the directory is resolved above, so a symlink sitting in an out/ could
# still name a target anywhere on disk and pass every check so far. Nothing
# generates one legitimately, so refuse rather than resolve: macOS ships no
# readlink -f to resolve it with portably.
if [ -L "$abs" ]; then die "refusing $abs -- it is a symlink" 5; fi

# Beside the input, so a worktree's slices stay with the worktree that made
# them instead of every branch writing over one shared directory.
outdir="$dir/sliced"
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
