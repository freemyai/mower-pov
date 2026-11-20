#!/usr/bin/env bash
# Four-device 1080p render with immutable identities and resumable frame pairs.
set -euo pipefail

if (( $# != 3 )); then
    printf 'Usage: %s RUN_BUNDLE BLENDER OUTPUT_DIR\n' "$0" >&2
    exit 2
fi
run=$1
blender=$2
out=$3
python=${MOWER_PYTHON:-.venv/bin/python}
mkdir -p "$out"
jobs=()
for gpu in {0..3}; do
    first=$((gpu * 300))
    "$python" -m mower_pov render "$run" --blender "$blender" --device OPTIX \
        --gpu-index "$gpu" --preset final --frames "$first:$((first + 300))" \
        --output "$out/batch-$gpu" &
    jobs+=("$!")
done
status=0
for job in "${jobs[@]}"; do wait "$job" || status=1; done
exit "$status"
