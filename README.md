# Sim2Real Benchmark

# Installation

## Simulator environment configuration

Though CityFlow and SUMO are stable on Windows and Linux systems, we recommend using Linux. This repo is tested to work on Linux. It currently does not work on Windows.

### CityFlow Environment

CityFlow version 0.1 is used for the experiments. To install the CityFlow simulator, please follow the instructions in the [CityFlow Doc](https://cityflow.readthedocs.io/en/latest/install.html#).

```
sudo apt update && sudo apt install -y build-essential cmake

git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow
pip install .
```

To test configuration:

```
import cityflow
env = cityflow.Engine
```

### SUMO Environment

SUMO version 1.26.0 is used for the experiments. To install the SUMO environment, please follow the instructions in the [SUMO Doc](https://epics-sumo.sourceforge.io/sumo-install.html#).

The following instructions should work for Ubuntu 24.04.

```
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

You also need to install libsumo as Python modules to integrate SUMO with Python.

```
pip install libsumo==1.26.0
```

To test configuration:

```
import libsumo
import traci
```

