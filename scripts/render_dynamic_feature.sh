#!/usr/bin/env bash
# Run the onboard review, course preflight, and full-film production commands.
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${MOWER_PYTHON:-$repo/.venv/bin/python}" "$repo/scripts/render_dynamic_feature.py" "$@"
