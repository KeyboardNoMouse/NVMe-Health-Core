# ⚡ NVMe Health Core: Predictive Telemetry Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge\&logo=scikit-learn\&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge\&logo=pandas\&logoColor=white)

An **enterprise-grade AI diagnostic engine and full-stack web interface** built to predict NVMe drive failures across **Lenovo ThinkSystem servers and ThinkPad devices** before they occur.

---

# 🎯 Executive Summary

Currently, NVMe failures are detected **reactively** — only after severe errors or complete hardware failure.

This project shifts the paradigm to **predictive maintenance**.

By systematically analyzing **device-level SMART telemetry** such as:

* Temperature
* Error rates
* Lifetime usage
* Read/write statistics

the machine learning engine identifies **recurring signatures of hardware degradation**.

This allows IT administrators to **replace high-risk drives before catastrophic failure**, preventing:

* Data loss
* Server downtime
* Infrastructure instability

---

# 🚀 Key Features

### 🧠 Random Forest Prediction Engine

A robust machine learning classifier trained to detect **complex, non-linear relationships** in NVMe telemetry data.

### ⚖️ Advanced Data Balancing (SMOTE)

The dataset had a **98% class imbalance**, where most drives were healthy.
SMOTE (Synthetic Minority Over-sampling Technique) was used to generate synthetic samples for rare failure classes such as:

* Firmware crashes
* Early-life manufacturing defects

### 📊 ThinkSystem Diagnostic Dashboard

A modern **Streamlit-based web dashboard** featuring:

* Interactive telemetry analysis
* Real-time failure predictions
* Lenovo-inspired UI styling
* Drive health risk indicators

---

# 🧠 Hardware Failure Patterns

Using **Exploratory Data Analysis (EDA)** and machine learning, the system identifies the following NVMe failure patterns:

### 🔴 Pattern 1 — Wear-Out Failure

Occurs when SSDs approach their write endurance limits.

Indicators:

* High **TBW (Terabytes Written)**
* High **Percent Life Used**
* Large **Power-On Hours**

---

### 🟠 Pattern 4 — Controller / Firmware Failure

Controller instability caused by firmware issues.

Indicators:

* High **read/write error spikes**
* Low or zero **media errors**
* Specific **firmware version correlations**

---

### 🟡 Pattern 5 — Early-Life Failure

Manufacturing defects appearing in the early lifetime of the drive.

Indicators:

* **Low power-on hours (<3000)**
* Rapid error accumulation
* Abnormally high failure probability

---

⚠️ **Note**

Pattern **2 (Thermal Failure)** and **3 (Power-Related Failure)** are supported by the system architecture but were **not present in the provided synthetic dataset**.

---

# 🧰 Tech Stack

| Component        | Technology           |
| ---------------- | -------------------- |
| Language         | Python               |
| Data Processing  | Pandas               |
| Machine Learning | Scikit-Learn         |
| Model            | Random Forest        |
| Dashboard        | Streamlit            |
| Visualization    | Matplotlib / Seaborn |

---

# 💻 Installation & Execution

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/nvme-health-core.git
cd nvme-health-core
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Train the Model (Backend)

```bash
python model_training.py
```

This step:

* preprocesses the dataset
* balances the data using **SMOTE**
* trains the **Random Forest classifier**
* saves the trained model

---

## 4️⃣ Launch the Dashboard (Frontend)

```bash
streamlit run app.py
```

The dashboard will open in your browser.

Features include:

* Drive health prediction
* Failure probability scoring
* SMART telemetry visualization

---

# 📊 Example Prediction Output

```
NVMe Drive Health Report

Drive Status: ⚠️ HIGH RISK

Failure Probability: 87%

Most Likely Cause:
Wear-Out Failure
```

---

# 📂 Project Structure

```
nvme-health-core
│
├── dataset
│   └── nvme_drive_dataset.csv
│
├── notebooks
│   └── exploratory_analysis.ipynb
│
├── model_training.py
├── app.py
├── requirements.txt
└── README.md
```


# ⭐ Future Improvements

* Real-time NVMe telemetry monitoring
* Integration with **Lenovo server management APIs**
* Deep learning anomaly detection
* Fleet-level predictive maintenance analytics
* Alert system for enterprise IT infrastructure

---

# 📜 License

This project is developed for **academic and research purposes**.
