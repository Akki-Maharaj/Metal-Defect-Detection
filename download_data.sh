#!/usr/bin/env bash
set -e

REPO_URL="https://github.com/abin24/Magnetic-tile-defect-datasets..git"
TMP_DIR="$(mktemp -d)"
OUT_DIR="data/full"

echo "Cloning dataset repository..."
git clone --depth 1 "$REPO_URL" "$TMP_DIR"

echo "Reorganizing into $OUT_DIR/<class>/ ..."
mkdir -p "$OUT_DIR"
for cls in MT_Blowhole MT_Break MT_Crack MT_Fray MT_Uneven MT_Free; do
    mkdir -p "$OUT_DIR/$cls"
    find "$TMP_DIR/$cls/Imgs" -type f \( -iname "*.jpg" -o -iname "*.png" \) \
        -exec cp {} "$OUT_DIR/$cls/" \;
done

rm -rf "$TMP_DIR"

echo "Done. Class counts:"
for d in "$OUT_DIR"/*/; do
    echo "  $d: $(ls "$d"/*.jpg 2>/dev/null | wc -l) images"
done

echo ""
echo "Now train with:  python -m src.train --data data/full"
