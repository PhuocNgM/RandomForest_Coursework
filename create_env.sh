#!/usr/bin/env bash
# create_env.sh - create a python venv with required dependencies
# Usage: ./create_env.sh [env-dir] [python-executable] [kernel-name] [requirements-file]
# Example: ./create_env.sh .venv python3 randomforest-coursework requirements.txt

set -euo pipefail

ENV_DIR="python_env"
PYTHON_BIN="python3"
KERNEL_NAME="randomforest-coursework"
REQ_FILE_ARG="requirements.txt"

echo "Using python: $PYTHON_BIN"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Error: $PYTHON_BIN not found in PATH." >&2
    exit 2
fi

if [ -d "$ENV_DIR" ]; then
    echo "Virtual environment directory '$ENV_DIR' already exists. Reusing."
else
    echo "Creating virtual environment at '$ENV_DIR'..."
    "$PYTHON_BIN" -m venv "$ENV_DIR"
fi

# shellcheck disable=SC1090
source "$ENV_DIR/bin/activate"

echo "Upgrading pip and installing build tools..."
pip install --upgrade pip setuptools wheel

# Resolve requirements file: accept absolute/relative path or search common locations
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANDIDATES=(
    "$REQ_FILE_ARG"
    "$SCRIPT_DIR/$REQ_FILE_ARG"
    "$SCRIPT_DIR/../$REQ_FILE_ARG"
    "$PWD/$REQ_FILE_ARG"
)

REQ_FILE=""
for c in "${CANDIDATES[@]}"; do
    if [ -n "$c" ] && [ -f "$c" ]; then
        REQ_FILE="$c"
        break
    fi
done

if [ -z "$REQ_FILE" ]; then
    echo "Error: No requirements file found (tried: ${CANDIDATES[*]})." >&2
    echo "Please provide a requirements file as the fourth argument or place one at a standard location." >&2
    exit 1
fi

echo "Installing dependencies from requirements file: $REQ_FILE"
pip install -r "$REQ_FILE"

# Install ipykernel and register the kernel
echo "Installing ipykernel and registering kernel '$KERNEL_NAME'..."
pip install ipykernel
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"

echo "Environment setup complete."
echo "To activate: source $ENV_DIR/bin/activate"
echo "To deactivate: deactivate"
