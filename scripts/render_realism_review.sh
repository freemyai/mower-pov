#!/usr/bin/env bash
# Matched static comparison and proxy contact samples. Uses four local OptiX devices.
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
render() {
    "$python" -m mower_pov render "$run" --blender "$blender" --device OPTIX --preset preview "$@"
}
render --gpu-index 0 --view showcase --yard house --frames 525:606 --output "$out/contact-0" &
a=$!
render --gpu-index 1 --view showcase --yard house --frames 606:687 --output "$out/contact-1" &
b=$!
render --gpu-index 2 --view onboard --yard house --frames 565:655 --output "$out/onboard" &
c=$!
(
    render --gpu-index 3 --view onboard --yard legacy --frames 595 --output "$out/static-fence"
    render --gpu-index 3 --view onboard --yard house --frames 595 --output "$out/static-house"
) &
d=$!
status=0
for job in "$a" "$b" "$c" "$d"; do wait "$job" || status=1; done
test "$status" -eq 0
"$python" -m mower_pov encode-preview "$out"/contact-{0,1} --frames 525:687 --fps 30 --output "$out/dog-contact-5p4s.mp4"
"$python" -m mower_pov encode-preview "$out/onboard" --frames 565:655 --fps 30 --output "$out/onboard-study-3s.mp4"
for video in "$out"/{dog-contact-5p4s,onboard-study-3s}.mp4; do
    ffmpeg -v error -xerror -i "$video" -map 0:v:0 -f null -
done
