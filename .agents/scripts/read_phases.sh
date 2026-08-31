#!/usr/bin/env bash
# Prints the YAML frontmatter of every phase file in assets/phases/.
# Run from the automodel/ directory:
#   bash scripts/read_phases.sh

PHASES_DIR="$(dirname "$0")/../assets/phases"

for file in "$PHASES_DIR"/*.md; do
    echo "=== $(basename "$file") ==="
    awk 'NR>1{ if(/^---$/) exit; print }' "$file"
    echo
done
