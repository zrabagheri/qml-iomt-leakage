# Identifier and Temporal Leakage in Privacy-Preserving Quantum ML for IoMT Intrusion Detection

Code for the paper *"Identifier and Temporal Leakage in Privacy-Preserving Quantum Machine Learning for Intrusion Detection in Healthcare IoT"* (submitted to ISDFS 2027).

## What is here
- `src/vqc.py` – exact NumPy statevector simulator of a data re-uploading variational quantum classifier (VQC), with parameter-shift and adjoint gradients
- `src/test_vqc.py` – checks the simulator against full matrix simulation and finite differences
- `src/test_adjoint.py` – checks adjoint gradients against the parameter-shift rule
- `src/grad_norms.py` – share of per-example gradients above the DP clipping bound
- `src/train.py` – Adam, FedAvg, DP-SGD and a Rényi-DP accountant for the sampled Gaussian mechanism
- `src/prep.py` – feature sets (identifier removal), blocked temporal folds and preprocessing
- `src/exp_leak2.py` – leakage analysis (Table I)
- `src/exp_main.py` – `E1` centralised comparison (Table II), `E2` protocol sensitivity (Fig. 2), `E3` federated + DP (Table III)
- `src/exp_e3b.py` – clipping vs noise ablation
- `src/tune.py` – VQC depth/learning-rate selection on the inner split (blocks 1–3 train, block 4 validation)
- `results/` – raw per-fold results; `figures/` – manuscript figures

## Data
Download `wustl-ehms-2020_with_attacks_categories.csv` (WUSTL-EHMS-2020, Hady et al., IEEE Access 2020) and place it at `data/wustl.csv`.

## Run
Run from the `src` folder with the data at `../data/wustl.csv` (or adjust the path in `prep.py`):
```bash
pip install -r requirements.txt
python test_vqc.py
python test_adjoint.py
python exp_leak2.py
python exp_main.py E1 && python exp_main.py E2 && python exp_main.py E3
python exp_e3b.py
python grad_norms.py
```
No quantum SDK is required; all circuits are simulated exactly.

## License
MIT
