# CIE6021 Final Project: Poisoning Attack Defense on Federated Learning Systems

The attack code is based on ESORICS 2020 paper: Data Poisoning Attacks Against Federated Learning Systems

## Setup

1) ```python3 generate_data_distribution.py``` This downloads the datasets, as well as generates a static distribution of the training and test data to provide consistency in experiments.
2) ```python3 generate_default_models.py``` This generates an instance of all of the models used in the paper, and saves them to disk.

## Label Flipping Attack / Filtering Defense
```python3 main.py```
