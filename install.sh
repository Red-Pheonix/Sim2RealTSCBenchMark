#!/bin/bash
# ============================================================================
# One-click environment setup (Linux). Installs into the CURRENTLY ACTIVE
# python environment (conda env or venv) -- activate one first, e.g.:
#
#   python3 -m venv .venv && source .venv/bin/activate
#   bash install.sh
#
# Steps: (1) python deps from requirements.txt, (2) SUMO python bindings
# (libsumo wheels bundle the simulator -- no system SUMO needed), (3) CityFlow
# built from source (needs gcc/g++ and cmake; on Debian/Ubuntu:
# `sudo apt install build-essential cmake`), (4) import smoke test.
# ============================================================================
set -e
cd "$(dirname "$0")"

# --- 0. sanity ---------------------------------------------------------------
if ! python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "ERROR: python >= 3.10 required (found: $(python --version 2>&1))."
  echo "Activate a suitable environment first, e.g.: python3 -m venv .venv && source .venv/bin/activate"
  exit 1
fi
echo "using $(python --version) at $(which python)"

# --- 1. python dependencies --------------------------------------------------
echo "[1/4] installing python dependencies (requirements.txt) ..."
pip install -r requirements.txt

# --- 2. SUMO bindings ----------------------------------------------------------
echo "[2/4] installing SUMO python bindings (libsumo + traci) ..."
pip install libsumo==1.26.0 traci==1.26.0

# --- 3. CityFlow (built from source) ------------------------------------------
echo "[3/4] building CityFlow ..."
if python -c 'import cityflow' 2>/dev/null; then
  echo "cityflow already installed, skipping build"
else
  for tool in cmake g++; do
    if ! command -v $tool > /dev/null; then
      echo "ERROR: '$tool' not found -- CityFlow builds from C++ source."
      echo "On Debian/Ubuntu: sudo apt update && sudo apt install -y build-essential cmake"
      exit 1
    fi
  done
  if [ ! -d CityFlow ]; then
    git clone --depth 1 https://github.com/cityflow-project/CityFlow.git
  fi
  pip install ./CityFlow
fi

# --- 4. smoke test -------------------------------------------------------------
echo "[4/4] verifying imports ..."
python - <<'EOF'
import cityflow
import libsumo
import traci
import torch
import pandas
print('cityflow OK  |  libsumo', libsumo.__version__ if hasattr(libsumo, '__version__') else 'OK',
      ' |  torch', torch.__version__)
EOF

echo
echo "Done. Try a first experiment (evaluates a pretrained policy, finishes in seconds):"
echo "  python run_s2r_actions.py -a dqn -n tempe_1x1 --act_model direct_transfer --real_setting setting2 --prefix my_run"
