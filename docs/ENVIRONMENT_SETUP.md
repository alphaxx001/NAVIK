# Environment Setup

This document records the baseline Python environment required to build and execute the initial ML and Data components of NAVIK. 

## Base Environment
* **Python Version:** 3.13.7

## Dependencies
The minimal dependencies required to run the current legacy data loaders and the AVNet PyTorch models are explicitly tracked to prevent over-pollution of the environment.

*   `torch` (PyTorch core for ML models)
*   `numpy` (Numerical operations and array handling)
*   `pandas` (CSV parsing and data manipulation for IO-VNBD datasets)

*Note: `scipy` is also available in the environment and will be utilized later for classical baseline filters, but is not strictly required by the current legacy ML scripts.*

## Installation Command
To bootstrap a new environment, run the following from the `IDR-System` root directory:

```bash
pip install -r requirements.txt
```

## Verification Command
To verify that the environment is correctly set up and that the repository structure resolves properly, run the following verification checks:

```bash
# Verify base libraries
python -c "import pandas as pd; import numpy as np; import torch; import scipy; print('Standard libraries OK')"

# Verify ML package structure
python -c "import ml.data.dataset_loader; import ml.models.speednet.avnet_legacy; import ml.training.train_legacy; print('ML package structure OK')"

# Verify legacy compatibility wrappers
python -c "import AI_Model.dataset_loader; print('Dataset loader wrapper OK')"
python -c "import AI_Model.avnet; print('AVNet wrapper OK')"
```

If all commands output "OK", the environment is successfully bootstrapped for Phase 1.
