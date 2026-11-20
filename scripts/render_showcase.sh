#!/usr/bin/env bash
# Four-device 720p render for the standalone 18-second garden montage.
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
ranges=("0:120,1090:1095" "525:660" "660:675,780:900" "900:930,975:1090")
jobs=()
for gpu in {0..3}; do
    "$python" -m mower_pov render "$run" --blender "$blender" --device OPTIX --gpu-index "$gpu" \
        --preset beauty --view showcase --frames "${ranges[gpu]}" --output "$out/batch-$gpu" &
    jobs+=("$!")
done
status=0
for job in "${jobs[@]}"; do wait "$job" || status=1; done
test "$status" -eq 0
"$python" "$(dirname "$0")/assemble_showcase.py" "$run" "$out"
