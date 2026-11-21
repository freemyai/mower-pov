#!/usr/bin/env bash
# Front door for the onboard review and full-film production workflow.
set -euo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${MOWER_PYTHON:-$repo/.venv/bin/python}" "$repo/scripts/render_dynamic_feature.py" "$@"
