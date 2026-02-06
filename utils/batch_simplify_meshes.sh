#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MESHLAB_SCRIPT="${MESHLAB_SCRIPT:-$SCRIPT_DIR/simplify_5000.mlx}"
MESHLABSERVER="${MESHLABSERVER:-meshlabserver}"
INPUT_DIR="${1:-.}"
SUFFIX="${2:-_low}"
INPLACE="${3:-}"

if ! command -v "$MESHLABSERVER" >/dev/null 2>&1; then
  echo "meshlabserver not found. Install MeshLab or set MESHLABSERVER=/path/to/meshlabserver" >&2
  exit 1
fi

if [[ ! -f "$MESHLAB_SCRIPT" ]]; then
  echo "MeshLab script not found: $MESHLAB_SCRIPT" >&2
  exit 1
fi

shopt -s nullglob
files=("$INPUT_DIR"/*.STL "$INPUT_DIR"/*.stl)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "No STL files found in $INPUT_DIR" >&2
  exit 1
fi

for f in "${files[@]}"; do
  if [[ "$f" == *".tmp."* ]]; then
    continue
  fi
  ext="${f##*.}"
  if [[ "$INPLACE" == "--inplace" ]]; then
    tmp="$(mktemp "${f%.*}.tmp.XXXXXX.${ext}")"
    echo "Simplifying (in-place): $f"
    "$MESHLABSERVER" -i "$f" -o "$tmp" -s "$MESHLAB_SCRIPT"
    mv -f "$tmp" "$f"
  else
    out="${f%.*}${SUFFIX}.${ext}"
    echo "Simplifying: $f -> $out"
    "$MESHLABSERVER" -i "$f" -o "$out" -s "$MESHLAB_SCRIPT"
  fi
done
