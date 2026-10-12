#!/bin/bash
# pack_code.sh - tarball of ONLY this repository's scripts and configs needed on the pod (no data, no weights; refuses above 50 MB).
#   bash tools/cloud/pack_code.sh [OUT_DIR]     (default OUT_DIR=$VC_WORK/cloud/dist, i.e. ~/vesuvius/work/cloud/dist)
# The exact file list comes from pack_manifest.txt; MANIFEST.tsv (path, bytes, sha256) is written next to the tarball and
# included in it as tools/cloud/MANIFEST.tsv (nothing is written into the repository). Paths are relative to the
# repository root (PACK_SRC), so on the pod:
#   mkdir -p CODE_DIR && tar -xzf vesuvius_code.tgz -C CODE_DIR
set -euo pipefail
SRC=${PACK_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
OUT=${1:-${VC_WORK:-${VC_ROOT:-$HOME/vesuvius}/work}/cloud/dist}
LIMIT_MB=${PACK_LIMIT_MB:-50}
mkdir -p "$OUT"
cd "$SRC"
shopt -s globstar nullglob
LIST=$(mktemp)
missing=0
while IFS= read -r line; do
  line=${line%%#*}; line=$(echo "$line" | xargs)      # strip comments / whitespace
  [ -z "$line" ] && continue
  matched=0
  for f in $line; do
    [ -f "$f" ] || continue
    case "$f" in
      *__pycache__*|*.pyc|*.bak*|*nosense_bak*|*.so|*/build/*|tools/cloud/dist/*|tools/cloud/tests/dryrun_*) continue ;;
    esac
    echo "$f" >> "$LIST"; matched=1
  done
  [ $matched = 1 ] || { echo "WARNING: manifest pattern matched nothing: $line" >&2; missing=$((missing + 1)); }
done < "$SRC/tools/cloud/pack_manifest.txt"
sort -u "$LIST" -o "$LIST"
MAN=$OUT/MANIFEST.tsv
{ printf 'path\tbytes\tsha256\n'; while IFS= read -r f; do printf '%s\t%s\t%s\n' "$f" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"; done < "$LIST"; } > "$MAN"
TGZ=$OUT/vesuvius_code.tgz
tar -czf "$TGZ.tmp" --owner=0 --group=0 -T "$LIST" -C "$OUT" MANIFEST.tsv --transform 's,^MANIFEST\.tsv$,tools/cloud/MANIFEST.tsv,'
mv "$TGZ.tmp" "$TGZ"
N=$(( $(wc -l < "$LIST") + 1 )); RAW=$(awk -F'\t' 'NR>1 {s+=$2} END {print s}' "$MAN"); SZ=$(stat -c %s "$TGZ")
sha256sum "$TGZ" | cut -d' ' -f1 > "$TGZ.sha256"
rm -f "$LIST"
echo "packed $N files, $((RAW / 1024)) KiB raw -> $TGZ ($((SZ / 1024)) KiB), sha256 $(cat "$TGZ.sha256")"
echo "by top-level dir:"; awk -F'\t' 'NR>1 {split($1, p, "/"); k=p[1] "/" p[2]; s[k]+=$2; n[k]++} END {for (k in s) printf "  %-28s %5d files %8d KiB\n", k, n[k], s[k]/1024}' "$MAN" | sort
[ "$missing" -eq 0 ] || echo "WARNING: $missing manifest patterns matched nothing" >&2
[ "$SZ" -le $((LIMIT_MB * 1024 * 1024)) ] || { echo "REFUSING: tarball $((SZ / 1048576)) MB > $LIMIT_MB MB"; exit 3; }
