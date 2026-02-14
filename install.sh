#!/usr/bin/env bash
set -euo pipefail

############################################################
# PXDesign One-Click Installation Script
#
# This script will:
#   1. Create a dedicated conda/mamba/micromamba environment
#   2. Install GPU PyTorch matching a specified CUDA version
#   3. Install Protenix
#   4. Install PXDesignBench dependencies
#   5. Clone PXDesign repo and install
#   6. Run basic import sanity checks
#
# Supported options:
#   --env <name>           Conda/mamba environment name (default: pxdesign)
#   --pkg_manager <tool>   conda | mamba | micromamba (default: conda)
#   --cuda-version <ver>   CUDA version string, e.g. 12.1, 12.2, 12.4
#                          Required. Must be >= 12.1.
############################################################

# Default configuration
env_name="pxdesign"
pkg_manager="conda"      # conda | mamba | micromamba
cuda_version=""          # e.g. 12.1, 12.2, 12.4

# Ensure script runs from its own directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ----------------------------------------------------------
# Parse command-line options
# ----------------------------------------------------------
OPTIONS=e:p:c:
LONGOPTIONS=env:,pkg_manager:,cuda-version:

PARSED=$(getopt --options="${OPTIONS}" --longoptions="${LONGOPTIONS}" --name "$0" -- "$@") || {
  echo "Error: failed to parse command line options."
  exit 1
}
eval set -- "${PARSED}"

while true; do
  case "$1" in
    -e|--env)
      env_name="$2"
      shift 2
      ;;
    -p|--pkg_manager)
      pkg_manager="$2"
      shift 2
      ;;
    -c|--cuda-version)
      cuda_version="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Invalid option: $1" >&2
      exit 1
      ;;
  esac
done

echo "=================================================="
echo " PXDesign Installation"
echo "   Environment name : ${env_name}"
echo "   Package manager  : ${pkg_manager}"
echo "   CUDA version     : ${cuda_version:-<not specified>}"
echo "=================================================="

SECONDS=0

############################################################
# CUDA version checks & PyTorch CUDA tag selection
############################################################

# Helper: check if version >= 12.1
check_cuda_ge_12_1() {
  local ver="$1"
  local major="${ver%%.*}"
  local rest="${ver#*.}"
  local minor="${rest%%.*}"

  if (( major > 12 )); then
    return 0
  elif (( major == 12 && minor >= 1 )); then
    return 0
  else
    return 1
  fi
}

if [ -z "${cuda_version}" ]; then
  echo "Error: --cuda-version must be specified (e.g., --cuda-version 12.1)."
  exit 1
fi

if ! check_cuda_ge_12_1 "${cuda_version}"; then
  echo "Error: CUDA version must be >= 12.1, but got '${cuda_version}'."
  exit 1
fi

# Decide PyTorch CUDA tag from CUDA version
# Extend this mapping as needed.
torch_tag=""
torch_version="2.3.1"   # adjust if needed
cuda_major="${cuda_version%%.*}"
cuda_minor="${cuda_version#*.}"
if [[ "${cuda_version}" == "${cuda_minor}" ]]; then
  cuda_minor="0"
else
  cuda_minor="${cuda_minor%%.*}"
fi

if (( cuda_major == 12 && cuda_minor >= 4 )) || (( cuda_major > 12 )); then
  torch_tag="cu124"
elif [[ "${cuda_version}" == 12.1* || "${cuda_version}" == 12.2* ]]; then
  torch_tag="cu121"
else
  echo "Error: unsupported CUDA version '${cuda_version}' for this installer."
  echo "       Currently supported: 12.1, 12.2 (cu121) and >=12.4 (cu124)."
  exit 1
fi

############################################################
# Package manager detection and initialization
############################################################

case "${pkg_manager}" in
  conda)
    if ! command -v conda >/dev/null 2>&1; then
      echo "Error: conda is not installed or not in PATH."
      exit 1
    fi
    env_tool="conda"
    ;;
  mamba)
    if ! command -v mamba >/dev/null 2>&1; then
      echo "Error: mamba is not installed or not in PATH."
      exit 1
    fi
    if ! command -v conda >/dev/null 2>&1; then
      echo "Error: mamba is installed but conda is not available."
      exit 1
    fi
    env_tool="mamba"
    ;;
  micromamba)
    if ! command -v micromamba >/dev/null 2>&1; then
      echo "Error: micromamba is not installed or not in PATH."
      exit 1
    fi
    env_tool="micromamba"
    ;;
  *)
    echo "Error: unsupported pkg_manager '${pkg_manager}'. Use 'conda', 'mamba', or 'micromamba'."
    exit 1
    ;;
esac

install_dir=$(pwd)
echo "Install root    : ${install_dir}"

############################################################
# Create and activate environment
############################################################

if [ "${env_tool}" = "micromamba" ]; then
  echo ">>> Using micromamba to manage environments"

  # Ensure MAMBA_ROOT_PREFIX is defined to avoid "unbound variable" under `set -u`
  export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"

  # Initialize micromamba shell hook for bash
  eval "$(micromamba shell hook -s bash)"

  echo ">>> Creating environment '${env_name}' (Python 3.11) with micromamba"
  micromamba create -y -n "${env_name}" python=3.11 || {
    echo "Error: failed to create environment ${env_name} with micromamba"
    exit 1
  }

  echo ">>> Activating environment '${env_name}' (micromamba)"
  micromamba activate "${env_name}" || {
    echo "Error: failed to activate environment ${env_name} with micromamba"
    exit 1
  }

else
  echo ">>> Using ${env_tool} to manage environments"

  CONDA_BASE=$(conda info --base 2>/dev/null) || {
    echo "Error: conda is not installed or cannot be initialized."
    exit 1
  }

  echo "Conda base      : ${CONDA_BASE}"

  echo ">>> Creating environment '${env_name}' (Python 3.11) with ${env_tool}"
  "${env_tool}" create -y -n "${env_name}" python=3.11 || {
    echo "Error: failed to create environment ${env_name} with ${env_tool}"
    exit 1
  }

  echo ">>> Activating environment '${env_name}' (${env_tool})"
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${env_name}" || {
    echo "Error: failed to activate environment ${env_name}"
    exit 1
  }

  if [ "${CONDA_DEFAULT_ENV:-}" != "${env_name}" ]; then
    echo "Error: expected environment '${env_name}' to be active, but got '${CONDA_DEFAULT_ENV:-}'."
    exit 1
  fi
fi

echo "Environment '${env_name}' successfully activated."
PYTHON_BIN="$(command -v python3 || command -v python)"
if [ -z "${PYTHON_BIN}" ]; then
  echo "Error: python executable not found after environment activation."
  exit 1
fi
echo "Python interpreter: ${PYTHON_BIN}"

if [ -n "${VIRTUAL_ENV:-}" ]; then
  echo "VIRTUAL_ENV: ${VIRTUAL_ENV}"
fi
echo "Conda env (if any): ${CONDA_DEFAULT_ENV:-<not-set>}"
echo "PYTHONPATH: ${PYTHONPATH:-<not-set>}"

############################################################
# Python package installation
############################################################

echo ">>> Upgrading pip"
"${PYTHON_BIN}" -m pip install --upgrade pip

# ----------------------------------------------------------
# 1) Install GPU PyTorch first (matching CUDA version)
# ----------------------------------------------------------
echo ">>> Installing PyTorch (GPU, CUDA ${cuda_version}, tag ${torch_tag})"
"${PYTHON_BIN}" -m pip install --no-cache-dir \
  "torch==${torch_version}" \
  --index-url "https://download.pytorch.org/whl/${torch_tag}" \
  || { echo "Error: failed to install PyTorch ${torch_version} with ${torch_tag} wheels."; exit 1; }

"${PYTHON_BIN}" - << 'PYTORCH_CHECK'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available :", torch.cuda.is_available())
print("Torch CUDA     :", torch.version.cuda)
if torch.cuda.is_available():
    print("CUDA devices   :", torch.cuda.device_count())
PYTORCH_CHECK

# ----------------------------------------------------------
# 2) Install Protenix & PXDesignBench
# ----------------------------------------------------------

echo ">>> Installing Protenix"
"${PYTHON_BIN}" -m pip install --no-cache-dir "git+https://github.com/bytedance/Protenix.git@v0.5.0+pxd" \
  || { echo "Error: failed to install Protenix."; exit 1; }

echo ">>> Installing PXDesignBench base dependencies"
"${PYTHON_BIN}" -m pip install --no-cache-dir \
  einops \
  natsort \
  dm-tree \
  posix_ipc \
  "transformers==4.51.3" \
  "dm-haiku==0.0.13" \
  "optax==0.2.5" \
  || { echo "Error: failed to install base Python dependencies."; exit 1; }

echo ">>> Installing ColabDesign (without dependencies)"
"${PYTHON_BIN}" -m pip install --no-cache-dir git+https://github.com/sokrypton/ColabDesign.git --no-deps \
  || { echo "Error: failed to install ColabDesign."; exit 1; }

echo ">>> Installing JAX with CUDA support"
"${PYTHON_BIN}" -m pip install --no-cache-dir \
  "jax[cuda]==0.4.29" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
  || { echo "Error: failed to install JAX (CUDA build)."; exit 1; }

# downgrade numpy
"${PYTHON_BIN}" -m pip install --no-cache-dir \
  "numpy==1.26.3" \
  || { echo "Error: failed to install numpy 1.26.3."; exit 1; }

echo ">>> Installing PXDesignBench (bundled)"
"${PYTHON_BIN}" -m pip install -e ./PXDesignBench \
  || { echo "Error: failed to install PXDesignBench."; exit 1; }

echo ">>> Installing PXDesign"
"${PYTHON_BIN}" -m pip install -e .

if [ "${env_tool}" = "micromamba" ]; then
  micromamba install -c conda-forge cudnn -y || { echo "Error: failed to install cudnn with micromamba."; exit 1; }
else
  conda install -c conda-forge cudnn -y || { echo "Error: failed to install cudnn with conda."; exit 1; }
fi


# -------------------------------
# 3) CUTLASS (for DeepSpeed Evo attention)
# -------------------------------

# Default to $HOME/cutlass if CUTLASS_PATH is not set by the user
export CUTLASS_PATH="${CUTLASS_PATH:-$HOME/cutlass}"

echo "[CUTLASS] Using CUTLASS_PATH=${CUTLASS_PATH}"

if [ ! -d "${CUTLASS_PATH}" ]; then
  echo "[CUTLASS] CUTLASS not found, cloning NVIDIA/cutlass v3.5.1 ..."
  git clone -b v3.5.1 https://github.com/NVIDIA/cutlass.git "${CUTLASS_PATH}"
else
  echo "[CUTLASS] Existing CUTLASS directory detected, skipping clone."
fi


############################################################
# Sanity checks
############################################################

echo ">>> Running sanity checks (import tests)"

export PYINSTALL_DIR="${SCRIPT_DIR}"
"${PYTHON_BIN}" - << 'PYCODE'
import sys
import os
import shutil
from pathlib import Path
print(f"PYTHON_EXECUTABLE={sys.executable}")

def check(mod):
    try:
        __import__(mod)
        print(f"[OK] import {mod}")
    except Exception as e:
        print(f"[FAIL] import {mod}: {e}", file=sys.stderr)
        raise

modules = [
    "torch",
    "jax",
    "jax.numpy",
    "colabdesign",
    "protenix",
    "pxdbench",
    "pxdesign"
]

for m in modules:
    check(m)

import jax
print("JAX devices:", jax.devices())

pxd_dir = Path(os.environ.get("PYINSTALL_DIR", "/root/PXDesign"))
assert pxd_dir.exists(), f"Expected PXDesign repo at {pxd_dir} missing"
print(f"[OK] PXDesign checkout found at {pxd_dir}")

pxdesign_bin = shutil.which("pxdesign")
if not pxdesign_bin:
    raise RuntimeError("pxdesign CLI is not available on PATH in the active interpreter")
print(f"[OK] pxdesign entry point: {pxdesign_bin}")
PYCODE

echo "Sanity checks completed."

############################################################
# Cleanup and final message
############################################################

echo ">>> Cleaning up package manager caches"

if [ "${env_tool}" = "micromamba" ]; then
  micromamba clean -a -y || echo "Warning: failed to clean micromamba caches."
  micromamba deactivate || true
else
  "${env_tool}" clean -a -y || echo "Warning: failed to clean ${env_tool} caches."
  conda deactivate || true
fi

t=${SECONDS}
echo "=================================================="
echo " PXDesign environment setup done!"
echo "   Environment name : ${env_name}"
echo "   Package manager  : ${pkg_manager}"
echo "   CUDA version     : ${cuda_version} (torch tag: ${torch_tag})"
echo
echo " Activate with:"
if [ "${env_tool}" = "micromamba" ]; then
  echo "   micromamba activate ${env_name}"
else
  echo "   conda activate ${env_name}"
fi
echo
echo " Installation time: $((t / 3600))h $(((t / 60) % 60))m $((t % 60))s"
echo "=================================================="
