# Privacy-Preserving Quantum Machine Learning Framework for IoMT Intrusion Detection

Code for the paper *"Privacy-Preserving Quantum Machine Learning Framework for Detecting Cyber Attacks in Smart Healthcare Systems"* by Omar Farshad Jeelani and Zahra Bagheri (submitted to ISDFS 2027).

The framework combines a compact data re-uploading variational quantum classifier (VQC), federated averaging (FedAvg) across sites and example-level differential privacy (DP-SGD with Rényi accounting). It is evaluated on WUSTL-EHMS-2020 with device identifiers removed and blocked temporal cross-validation.

## What is here
- `src/vqc.py` – exact NumPy statevector simulator of the VQC, with parameter-shift and adjoint gradients
- `src/test_vqc.py` – checks the simulator against full matrix simulation and finite differences
- `src/test_adjoint.py` – checks adjoint gradients against the parameter-shift rule
- `src/train.py` – Adam, FedAvg, DP-SGD and a Rényi-DP accountant for the sampled Gaussian mechanism
- `src/prep.py` – feature sets (identifier removal), blocked temporal folds and preprocessing
- `src/exp_main.py` – `E1` quantum vs classical comparison (Table I), `E2` protocol sensitivity, `E3` federated + DP (Table II, Fig. 2)
- `src/exp_e3b.py` – clipping vs noise ablation
- `src/grad_norms.py` – share of per-example gradients above the DP clipping bound
- `src/exp_leak2.py` – preliminary identifier/temporal analysis described in Section III-B
- `src/tune.py` – VQC depth/learning-rate selection on the inner split (blocks 1–3 train, block 4 validation)
- `src/tune_broad.py`, `src/tune_broad_outer.py` – broader search on the inner split (up to 8 layers, 30 epochs) and its outer-fold check
- `results/` – raw per-fold results; `figures/` – figures (`Fig1_circuit.png` = Fig. 1, `Fig3_dp.png` = Fig. 2 in the paper; `Fig2_leakage.png` is supplementary)

## Data
Download `wustl-ehms-2020_with_attacks_categories.csv` (WUSTL-EHMS-2020, Hady et al., IEEE Access 2020) and place it at `data/wustl.csv`.

## Run
Run from the `src` folder with the data at `../data/wustl.csv`:
```bash
pip install -r requirements.txt
python test_vqc.py
python test_adjoint.py
python exp_leak2.py
python exp_main.py E1 && python exp_main.py E2 && python exp_main.py E3
python exp_e3b.py
python grad_norms.py
python tune_broad.py && python tune_broad_outer.py
```
No quantum SDK is required; all circuits are simulated exactly.

## License
MIT
