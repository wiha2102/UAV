#!/usr/bin/env bash

# Exit immediately if:
#   - a command exits with non-zero status (-e)
#   - an undefined variable is being used (-u)
#   - a pipeline fails anywhere (-o pipefail)
set -euo pipefail

# Name of the virtual environment (.venv) would been better :/
VENV="venv"

# Python executable that default at python3.10 (python3.10.16) 
# in my case, python3.10 is reliable and preferable 
PYTHON="${1:-python3.10}"


# ------------------------------------------------------------
#   Utility Functions
# ------------------------------------------------------------

# Checks if a command exist in the PATH
command_exists() {
    command -v "$1" &>/dev/null
}

# Prints the error message and dies (exist the installation)
die() {
    echo "[Error]: $1" >&2
    exit 1
}

# Print the informational message
info() {
    echo "[Info]: $1"
}


# ------------------------------------------------------------
#   System & Dependency Checks  
# ------------------------------------------------------------

# Ensure required system tools are available
command_exists "$PYTHON" || die "$PYTHON not found (Python 3.10 required)"
command_exists git || die "git is required but not installed"
command_exists nvidia-smi || die "nvidia-smi not found (NVIDIA driver required)"

# Enforce Python 3.10 (Tensorflow 2.20 GPU Compatibility constraint)
PY_VER="$("$PYTHON" - <<EOF
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
EOF
)"
[[ "$PY_VER" == "3.10" ]] || die "Python 3.10 required, found $PY_VER"


# ------------------------------------------------------------
#   Creation of the Virtual Environment
# ------------------------------------------------------------

# If virtual environment already exist, ask if recreate to avoid remake all if not
# necessary, should only if not recreate only proceed installation of uninstalled 
# packages (not tested could be buggy)
if [ -d "$VENV" ]; then
    echo "[Warning]: Virtual environment '$VENV' already exists."
    read -rp "Remove and recreate it? [y/N]: " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        info "Removing existing virtual environment..."
        rm -rf "$VENV"
    else
        info "Reusing existing virtual environment."
    fi
fi

# Create the virtual environment if not already exist
if [ ! -d "$VENV" ]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV"
fi

# Activate venv
# shellcheck disable=SC1090
source "$VENV/bin/activate"


# ------------------------------------------------------------
#   Core Tooling
# ------------------------------------------------------------

# Ensure pip exists and upgrade core packaging tools
info "Upgrading pip tooling..."
python -m ensurepip --upgrade || true
pip install --upgrade pip setuptools wheel


# ------------------------------------------------------------
#   Tensorflow (Version depending on CUDA version for GPU support)
# ------------------------------------------------------------

# Clean possible previous installations due to Nvidia sionna installation may
# not support the GPU 
info "Installing TensorFlow with CUDA support..."
pip uninstall -y tensorflow tensorflow-cpu tensorflow-intel &>/dev/null || true

# Specifically install tensorflow 2.20.x (compatible with CUDA 13)
pip install "tensorflow[and-cuda]==2.20.*"

# Verify tensorflow is detecting GPU correctly to ensure it works in time
info "Verifying TensorFlow GPU visibility..."

python - <<'EOF'
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("Detected GPUs:", gpus)

if not gpus:
    raise RuntimeError("TensorFlow GPU NOT detected. Aborting.")
EOF

# ------------------------------------------------------------
#   Additional Libraries useful in this project
# ------------------------------------------------------------

# Install commonly used data science and performance libraries
info "Installing base dependencies..."
pip install pandas seaborn scikit-learn tqdm orjson pyarrow numba matplotlib



# ------------------------------------------------------------
#   Sionna Library downloaded using git at address
# ------------------------------------------------------------

# Install Sionna only and only if not already installed.
if ! pip show sionna &>/dev/null; then
    info "Installing Sionna from GitHub..."
    pip install git+https://github.com/NVlabs/sionna.git@main
else
    info "Sionna already installed."
fi


# ------------------------------------------------------------
#   Creating a requirements script for reproducibility
# ------------------------------------------------------------

# Optionally update requirements file after executing updates
read -rp "Update requirements.txt (excluding TensorFlow)? [Y/n]: " update_requirements
if [[ ! "$update_requirements" =~ ^[Nn]$ ]]; then
    info "Updating requirements.txt (TensorFlow excluded)..."
    pip freeze | grep -v -E '^(tensorflow|nvidia-)' > requirements.txt
fi


# ------------------------------------------------------------
#   All Done
# ------------------------------------------------------------

info "Setup completed successfully!"
deactivate
