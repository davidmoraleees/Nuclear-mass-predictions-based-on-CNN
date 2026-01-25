# Nuclear mass predictions based on Convolutional Neural Networks (CNN)

## Author
This project was created by [David Morales](https://www.linkedin.com/in/david-morales-361b41282/).

---

## License

This project is licensed under the Apache License 2.0.
See the LICENSE file for details.

---

## Brief summary

This repository contains the full pipeline developed for predicting nuclear binding energies and nuclear masses using convolutional neural networks (CNNs). The project combines traditional nuclear models (Liquid Drop Model and WS4) with data-driven deep learning approaches (CNN-I3 and CNN-I4) and provides tools for preprocessing, training, evaluation, and physics-oriented analysis.

---

## 1. Scientific Overview

### Physical problem
The goal is to predict:
- Total binding energy
- Nuclear mass

for atomic nuclei characterized by proton number \( Z \) and neutron number \( N \).

### Models implemented
- **Liquid Drop Model (LDM)**  
  Used as a baseline theoretical model.
- **WS4 model**  
  Used as a high-quality phenomenological reference.
- **CNN-I3**  
  CNN with 3 input channels:
  - \( Z \)-grid
  - \( N \)-grid
  - Local binding-energy neighborhood
- **CNN-I4**  
  CNN with 4 input channels:
  - Same as CNN-I3
  - Plus pairing indicator \( \delta \)

The CNNs operate on 5×5 neighborhoods in the nuclear chart, embedding local nuclear structure information.

---

## 2. Installation and Environment

### Requirements
- Python ≥ 3.12
- Poetry for dependency management

### Install dependencies
```bash
pip install poetry
poetry install
```

---

## 3. Configuration
All numerical constants, hyperparameters, plotting options, and paths are centralized in:
```bash
config.yaml
```

This includes:
- Physical constants (LDM parameters, electron mass, etc.)
- Dataset bounds in \( N \) and \( Z \).
- Training hyperparameters
- Random seed for reproducibility
- Plotting styles and image formats

Configuration loading is centralized via:
```bash
from src.utils.config import load_config
```

---

## 4. Pipeline Overview

### Step 1 — Data preprocessing
Parses raw AME and WS4 datasets, computes derived physical quantities, and produces cleaned CSV files.

```bash
poetry run preprocessing
```

Outputs:
- mass2016_cleaned.csv
- mass2020_cleaned.csv
- WS4_cleaned.csv
- Diagnostic plots comparing LDM and WS4 with experimental values.

### Step 2 — Model training
Train CNN-I3 or CNN-I4 on the processed data.

```bash
poetry run training --model_type I3 --outputs_folder outputs_I3
poetry run training --model_type I4 --outputs_folder outputs_I4
```

During training:
- Train/test split is reproducible via fixed random seed
- Early stopping is applied
- Best model weights are saved
- Training evolution plots are generated
- Nuclear-mass difference maps are produced


### Step 3 — Model evaluation
Evaluate trained models on AME 2016 and 2020 datasets, including subsets such as newly measured nuclei.

```bash
poetry run evaluate
```

Outputs:
- RMSE statistics
- Per-model prediction columns added to CSVs
- Difference maps
- Combined comparison plots (CNN vs experimental values)

### Step 4 — Isotopic and isotonic chains
Physics-driven evaluation along selected nuclear chains.

```bash
poetry run iso_chains
```

Produces:
- Isotopic chain plots
- Isotonic chain plots
- Training vs extrapolation distinction
- Comparison with LDM and WS4

## 5. Project report
For a full explanation of the methods and steps followed, refer to the `docs/report.pdf` file, which contains the detailed scientific paper on my approach.
