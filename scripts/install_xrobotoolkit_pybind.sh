#!/usr/bin/env bash
# Install XRoboToolkit Python bindings in an isolated Conda environment.

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYBIND_DIR="${ROOT_DIR}/external/XRoboToolkit-PC-Service-Pybind"
ENV_NAME="${XROBOTOOLKIT_ENV_NAME:-env_xrobotoolkit}"
PC_SERVICE_DIR="${XROBOTOOLKIT_PC_SERVICE_DIR:-/opt/apps/roboticsservice}"

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
    echo "Conda initialization script not found." >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

conda activate "${ENV_NAME}"
conda install -c conda-forge cmake make cxx-compiler pybind11 libstdcxx-ng -y
python -m pip install --upgrade pip setuptools wheel

if [[ ! -d "${PYBIND_DIR}/.git" ]]; then
    git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git "${PYBIND_DIR}"
fi

cd "${PYBIND_DIR}"

if [[ -f "${PC_SERVICE_DIR}/SDK/include/PXREARobotSDK.h" && -f "${PC_SERVICE_DIR}/SDK/x64/libPXREARobotSDK.so" ]]; then
    echo "Using installed XRoboToolkit PC Service SDK from: ${PC_SERVICE_DIR}"
    mkdir -p include lib
    cp -f "${PC_SERVICE_DIR}/SDK/include/PXREARobotSDK.h" include/
    cp -f "${PC_SERVICE_DIR}/SDK/x64/libPXREARobotSDK.so" lib/
    if [[ -d "${PC_SERVICE_DIR}/SDK/include/nlohmann" && ! -d include/nlohmann ]]; then
        cp -r "${PC_SERVICE_DIR}/SDK/include/nlohmann" include/nlohmann
    fi
else
    echo "Installed PC Service SDK not found, falling back to upstream setup_ubuntu.sh."
    bash setup_ubuntu.sh
fi

python -m pip uninstall -y xrobotoolkit_sdk || true
python -m pip install .

cat <<EOF

Installed xrobotoolkit_sdk in Conda environment: ${ENV_NAME}

Before running data tests, use:
  conda activate ${ENV_NAME}
  export LD_LIBRARY_PATH="${PYBIND_DIR}/lib:\${CONDA_PREFIX}/lib:\${LD_LIBRARY_PATH:-}"

EOF
