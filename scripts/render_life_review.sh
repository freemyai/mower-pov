#!/usr/bin/env bash
# Render the two animal review windows from the full absolute work timeline.
set -euo pipefail

if (( $# != 3 )); then
    printf 'Usage: %s RUN_BUNDLE BLENDER NEW_OUTPUT_DIR\n' "$0" >&2
    exit 2
fi
run=$1
blender=$2
out=$3
python=${MOWER_PYTHON:-.venv/bin/python}
mkdir "$out"
ranges=("0:60,525:555" "60:120,555:585" "585:675" "675:765")
jobs=()
for i in {0..3}; do
    "$python" -m mower_pov render "$run" --blender "$blender" --device OPTIX --gpu-index "$i" \
        --preset beauty --frames "${ranges[i]}" --output "$out/batch-$i" &
    jobs+=("$!")
done
status=0
for job in "${jobs[@]}"; do wait "$job" || status=1; done
test "$status" -eq 0

"$python" -m mower_pov encode-preview "$out"/batch-{0,1} --frames 0:120 --fps 30 --output "$out/hedgehog-4s.mp4"
"$python" -m mower_pov encode-preview "$out"/batch-{0,1,2,3} --frames 525:765 --fps 30 --output "$out/dog-and-return-8s.mp4"
videos=("$out"/hedgehog-4s.mp4 "$out"/dog-and-return-8s.mp4)
for video in "${videos[@]}"; do
    ffmpeg -v error -xerror -i "$video" -map 0:v:0 -f null -
done
sha256sum "${videos[@]}" > "$out/video.sha256"
