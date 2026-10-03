# AI Circuit Fault Detector

An ECE project that simulates a Sallen–Key band-pass circuit in LTspice, extracts signal features from transient waveforms, and compares a Random Forest with a 1D convolutional neural network.

> **Research prototype:** this repository is for simulation and education. The circuit model and classifier have not been validated on physical hardware. Do not connect it to live equipment or use predictions for safety or control decisions. The six fault classes are modelled approximations; validate against the actual op-amp, component parasitics, and measured circuits before making engineering claims.

## Circuit under test

The reference design cascades a first-order RC high-pass input with a second-order Sallen–Key low-pass stage. Its nominal component values are in [`circuits/sallen_key_bandpass.inc`](circuits/sallen_key_bandpass.inc); [`circuits/sallen_key_bandpass.asc`](circuits/sallen_key_bandpass.asc) is the LTspice entry point and [`circuits/sallen_key_bandpass.cir`](circuits/sallen_key_bandpass.cir) is the healthy netlist.

| Code | Label | Circuit intervention |
| --- | --- | --- |
| F0 | Healthy | Nominal circuit; healthy components vary by ±10% |
| F1 | R1-open | R1 replaced by 1 TΩ leakage |
| F2 | R2-short | R2 replaced by 1 mΩ |
| F3 | C1-open | C1 replaced by 1 fF leakage |
| F4 | C2-short | A 1 mΩ path bypasses C2 |
| F5 | Opamp-degraded | Behavioral op-amp open-loop gain reduced to 20 and output resistance raised to 50 kΩ |

The op-amp is a bounded behavioral model to keep the simulation self-contained. It is not a vendor macromodel. Inspect and adapt the deck if the project plan requires a specific amplifier part number.

## Requirements

- Python 3.10 or newer
- Analog Devices LTspice installed
- Git access to this private repository

## Install on Windows

```powershell
git clone https://github.com/kshitij1119/ai-circuit-fault-detector.git
cd ai-circuit-fault-detector
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

The `all` extra includes PyLTSpice, Streamlit, plotting dependencies, and PyTorch. If you only need the Random Forest workflow, `python -m pip install -e ".[simulation,app]"` installs the simulator and dashboard without PyTorch. LTspice is discovered from common Windows install locations; if it is installed elsewhere, pass `--ltspice` or set the `LTSPICE_EXE` environment variable.

## Run the pipeline

First run six simulations to check that LTspice launches correctly:

```powershell
python simulate.py --ltspice "C:\Users\ksp54\AppData\Local\Programs\ADI\LTspice\LTspice.exe" --limit 6
```

Then generate the planned dataset of 1,200 transients (200 tolerance samples per fault class):

```powershell
python simulate.py --samples-per-class 200 --seed 42
python extract_features.py
python train.py
python train_cnn.py
python eda.py
```

The simulation output is written to `data/ltspice/`: a manifest, one time/input/output CSV per run, and LTspice RAW/LOG files. `extract_features.py` produces `data/processed/features.csv`. The RF and CNN metrics and plots are written under `reports/`; the RF model is saved in `models/`. The repository includes six small LTspice demo waveforms, a pretrained RF model, and the RF/EDA report outputs so the app can run immediately after installation. Full generated datasets, LTspice RAW/LOG files, and CNN checkpoints remain local because they are reproducible outputs.

### Waveform features

The feature model uses the eight planned features: peak amplitude, RMS, 10–90% rise time, settling time, total harmonic distortion (THD), zero-crossing count, standard deviation, and skewness. Timing features are measured on the first rising edge and full-cycle envelopes; the definitions are documented in [`circuit_fault_detector/features.py`](circuit_fault_detector/features.py). THD uses the 1 kHz input fundamental by default; use `--fundamental-hz` if you change the stimulus.

### Streamlit demo

Run the dashboard:

```powershell
streamlit run streamlit_app.py
```

Demo Mode selects one of six included LTspice transients and uses the pretrained RF model. CSV upload accepts `time_s` and `output_v` columns. The app plots the waveform, displays all eight features, and shows predicted class probabilities.

## EDA and evaluation

`notebooks/eda.ipynb` and `eda.py` plot representative traces, feature distributions, and feature correlations. The RF uses a stratified 80/20 hold-out split and reports accuracy, precision, recall, F1, confusion matrix, and feature importance. The CNN uses stratified train/validation/test splits, selects the checkpoint by validation loss, and reports test metrics. Reference results from the included LTspice-generated dataset are listed below.

### Reference run

The checked-in report snapshot was generated from 1,200 LTspice simulations (200 per class, seed 42). Both models used the same stratified 20% test split. The RF used 500 trees; the CNN ran for five epochs for this reference run.

| Model | Held-out accuracy | Evaluation |
| --- | ---: | --- |
| Random Forest, eight extracted features | 100.0% | [`reports/random_forest_metrics.json`](reports/random_forest_metrics.json) |
| 1D-CNN, raw waveforms | 89.2% | [`reports/cnn_metrics.json`](reports/cnn_metrics.json) |

These are simulation-only results for this circuit model and seeded dataset. The perfect RF score indicates the modeled fault classes are readily separable under these assumptions; it is not evidence of performance on measured hardware.

## Repository layout

```text
circuit_fault_detector/  faults, waveform features, simulation, RF model, CLI
circuits/                LTspice reference deck and fault definitions
data/                    Six included demo waveforms; full dataset is local
models/                  Pretrained RF model; CNN checkpoints are local
notebooks/               EDA notebook
reports/                 RF evaluation metrics and EDA figures
simulate.py              LTspice batch dataset generator
extract_features.py      Waveform feature extraction
train.py                 Random Forest training/evaluation
train_cnn.py             Raw-waveform 1D-CNN training/evaluation
streamlit_app.py         Local web demo
```

## License

MIT. See [LICENSE](LICENSE).
