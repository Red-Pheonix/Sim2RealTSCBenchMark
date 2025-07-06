# Joint-Local Grounded Action Transformation for Sim-to-Real Transfer in Multi-Agent Traffic Control

## 🚦 Overview

This repository builds upon the UGAT codebase (forked from [UGAT](https://github.com/DaRL-LibSignal/UGAT)) to support our proposed **Joint-Local Grounded Action Transformation (JL-GAT)** framework. Our approach focuses on improving sim-to-real transfer in multi-agent traffic signal control by allowing flexible control over centralized, decentralized, and hybrid grounding configurations.

## 🔧 How to Run JL-GAT

To run the JL-GAT experiment, first follow the instructions section in [UGAT](https://github.com/DaRL-LibSignal/UGAT) to download the necessary resources (CityFlow & SUMO), then use the following command:

```bash
python run_s2r.py --network cityflow1x3 --agent presslight
```

You can swap out the network name with:

- `cityflow1x3` for the 1×3 network  
- `cityflow4x4` for the 4×4 network

This will initiate training and evaluation under the specified environment.

## ⚙️ Configuring JL-GAT

To customize the GAT configuration:

Open the config file:

```bash
JL-GAT/configs/tsc/base.yml
```

Modify the following parameters:

- `gattype`: Choose from:
  - `"centralized"`
  - `"decentralized"`
  - `"jlgat"`
- `gat`: Set to `false` to run without GAT.
- `prob_grounding`: Set a value (e.g., `0.2`) to control the probability of applying grounding during training.
