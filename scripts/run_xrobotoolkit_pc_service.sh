#!/usr/bin/env bash
# Start XRoboToolkit PC Service with a newer libstdc++ from Conda.

set -euo pipefail

SERVICE_DIR="${XROBOTOOLKIT_SERVICE_DIR:-/opt/apps/roboticsservice}"
CONDA_LIB="${XROBOTOOLKIT_CONDA_LIB:-${CONDA_PREFIX:-${HOME}/anaconda3}/lib}"

if [[ ! -x "${SERVICE_DIR}/RoboticsServiceProcess" ]]; then
    echo "RoboticsServiceProcess not found or not executable: ${SERVICE_DIR}/RoboticsServiceProcess" >&2
    exit 1
fi

if [[ ! -e "${CONDA_LIB}/libstdc++.so.6" ]]; then
    echo "Conda libstdc++.so.6 not found: ${CONDA_LIB}/libstdc++.so.6" >&2
    echo "Set XROBOTOOLKIT_CONDA_LIB to a directory containing libstdc++.so.6." >&2
    exit 1
fi

if ! strings "${CONDA_LIB}/libstdc++.so.6" | grep -F -x "GLIBCXX_3.4.32" >/dev/null; then
    echo "Conda libstdc++.so.6 does not provide GLIBCXX_3.4.32: ${CONDA_LIB}/libstdc++.so.6" >&2
    exit 1
fi

export LD_PRELOAD="${CONDA_LIB}/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
export LD_LIBRARY_PATH="${SERVICE_DIR}:${SERVICE_DIR}/lib:${SERVICE_DIR}/SDK/x64:${LD_LIBRARY_PATH:-}"
export QT_PLUGIN_PATH="${SERVICE_DIR}/plugins/:${QT_PLUGIN_PATH:-}"
export QT_QML_PATH="${SERVICE_DIR}/qml/:${QT_QML_PATH:-}"

echo "Starting XRoboToolkit PC Service from ${SERVICE_DIR}"
echo "Preloading libstdc++ from ${CONDA_LIB}"

cd "${SERVICE_DIR}"
exec ./RoboticsServiceProcess
