# MSTC-IDS: Multi-Scale Temporal Convolutional Intrusion Detection System for UAVs

A deep learning-based **Cyber-Physical Intrusion Detection System (IDS)** for **Unmanned Aerial Vehicles (UAVs)**, built on the **MSTC** (Multi-Scale Temporal Convolutional) architecture. The model detects cyber-physical attacks targeting UAV communication and control channels by learning multi-scale temporal patterns from flight telemetry and network data.

---

## 📄 Reference Paper

This work is based on the following publication:

> S. C. Hassler, U. A. Mughal and M. Ismail, **"Cyber-Physical Intrusion Detection System for Unmanned Aerial Vehicles,"** in *IEEE Transactions on Intelligent Transportation Systems*, vol. 25, no. 6, pp. 6106–6117, June 2024, doi: [10.1109/TITS.2023.3339728](https://doi.org/10.1109/TITS.2023.3339728).

Please cite the original paper if you use this dataset or build upon this work.

---

## 📊 Dataset

The dataset used in this repository is derived from the UAV cyber-physical intrusion dataset released alongside the paper above.

- **Source:** Hassler et al., IEEE T-ITS, 2024.
- **Format:** Segmented time-series data, split by attack type.
- **Location:** `segment_data/` folder in this repository.

The `segment_data/` directory contains the raw data segmented by attack class, allowing per-attack training, evaluation, and analysis (e.g., detecting replay attacks, GPS spoofing, DoS, etc.).

> ⚠️ If you are the dataset owner and would like it removed or attributed differently, please open an issue.

---

## 🧠 Model Overview
The MSTC (Multi-Scale Temporal Convolutional) network extracts features at multiple temporal resolutions using parallel convolutional branches with different kernel sizes. This allows the model to capture both:

Short-term anomalies (e.g., sudden packet drops, spoofed GPS jumps)

Long-term patterns (e.g., gradual drift, sustained replay behavior)

The extracted multi-scale features are fused and passed through dense layers for binary/multi-class classification (benign vs. specific attack types).

---

## 📈 Results
Evaluation metrics reported per attack segment typically include:

Metric	Description
Accuracy	Overall correct predictions
Precision	True positives / predicted positives
Recall	True positives / actual positives
F1-Score	Harmonic mean of precision & recall
AUC-ROC	Area under ROC curve

---

## ⚙️ Installation

# Clone the repository
git clone https://github.com/dradilhussain/MSTC_IDS-for-Cyber-Physical-Intrusion-Detection-in-Unmanned-Aerial-Vehicles.git
cd MSTC_IDS-for-Cyber-Physical-Intrusion-Detection-in-Unmanned-Aerial-Vehicles

# (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

🚀 Usage
1. Prepare the data
Ensure the segment_data/ folder contains the attack-segmented data. Update paths in config.yaml if needed.

2. Train the MSTC-IDS model
python train.py --config config.yaml

3. Evaluate on a specific attack segment
python evaluate.py --config config.yaml --attack <attack_name>

4. Inference on new data
python evaluate.py --config config.yaml --input path/to/segment --checkpoint path/to/model.pth

