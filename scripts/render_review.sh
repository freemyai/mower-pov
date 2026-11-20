#!/usr/bin/env bash
# Render the default backyard review package, optionally using four OptiX devices.
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  printf 'Usage: %s RUN_BUNDLE BLENDER NEW_OUTPUT_DIR [CPU|OPTIX|OPTIX4]\n' "$0" >&2
  exit 2
fi
run=$1
blender=$2
out=$3
mode=${4:-OPTIX}
python=${MOWER_PYTHON:-.venv/bin/python}
preset=${MOWER_REVIEW_PRESET:-preview}

case "$mode" in
  CPU|OPTIX) backend=$mode; cards=(0 0 0 0) ;;
  OPTIX4) backend=OPTIX; cards=(0 1 2 3) ;;
  *) printf 'Unknown mode %s\n' "$mode" >&2; exit 2 ;;
esac
mkdir "$out"

render() {
  "$python" -m mower_pov render "$run" --blender "$blender" --device "$backend" --preset "$preset" "$@"
}

if [[ "$mode" == OPTIX4 ]]; then
  pids=()
  for shard in {0..3}; do
    first=$((525 + 60 * shard))
    render --gpu-index "${cards[shard]}" --frames "$first:$((first+60))" --output "$out/clip-$shard" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  test "$status" -eq 0
else
  render --frames 525:765 --output "$out/clip-0"
fi

keyframes=108,315,438,510,582,702,738
bench=108,114,123,132,141,150,210,219,228,237,246,255,315,324,333,342,351,360,414,426,438,450,462,474,486,498,510,522,870,900,930,960,990,1020,1050,1080
if [[ "$mode" == OPTIX4 ]]; then
  render --gpu-index 0 --frames "$bench" --output "$out/bench" &
  b=$!
  render --gpu-index 1 --frames "$keyframes" --undenoised --output "$out/raw-png" &
  p=$!
  render --gpu-index 2 --frames "$keyframes" --undenoised --exr --output "$out/raw-exr" &
  e=$!
  render --gpu-index 3 --frames "$keyframes" --output "$out/keyframes" &
  k=$!
  status=0
  for pid in "$b" "$p" "$e" "$k"; do wait "$pid" || status=1; done
  test "$status" -eq 0
else
  render --frames "$bench" --output "$out/bench"
  render --frames "$keyframes" --undenoised --output "$out/raw-png"
  render --frames "$keyframes" --undenoised --exr --output "$out/raw-exr"
  render --frames "$keyframes" --output "$out/keyframes"
fi
"$python" -m mower_pov encode-preview "$out"/clip-* --frames 525:765 --fps 30 --output "$out/mower-pov-preview-8s.mp4"
"$python" scripts/review_preview.py "$out"
ffmpeg -hide_banner -v error -xerror -i "$out/mower-pov-preview-8s.mp4" -map 0:v:0 -f null -
sha256sum "$out/mower-pov-preview-8s.mp4" > "$out/video.sha256"
