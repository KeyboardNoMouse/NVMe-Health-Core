# ⚡ Lenovo NVMe Health Core: Predictive Telemetry Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

An enterprise-grade AI diagnostic engine and full-stack web interface built to predict NVMe drive failures across Lenovo ThinkSystem servers and ThinkPad devices before they occur.

## 🎯 Executive Summary
Currently, NVMe failures are detected reactively—only after severe errors or hardware bricking. This project shifts the paradigm to **predictive maintenance**. By systematically analyzing device-level SMART telemetry (temperature, error rates, lifetime usage), this machine learning engine identifies the recurring signatures of hardware degradation. This allows IT administrators to replace high-risk drives safely before experiencing catastrophic data loss or system downtime.

## 🚀 Key Features
* **Random Forest Engine:** A robust classification model trained to identify complex, non-linear relationships in raw hardware telemetry.
* **Advanced Data Balancing (SMOTE):** Engineered synthetic telemetry data to overcome a 98% class imbalance, allowing the model to successfully detect extremely rare firmware crashes and early-life manufacturing defects.
* **ThinkSystem Dashboard:** A sleek, interactive web interface built with Streamlit, featuring custom Lenovo-branded CSS and real-time fleet diagnostic metrics.

## 🧠 Hardware Failure Patterns
Based on the provided synthetic dataset representing real-world NVMe SMART logs, we performed Exploratory Data Analysis (EDA) and trained the model to isolate the following distinct anomaly clusters:
* **Pattern 1 (Wear-Out Failure):** Drives nearing end-of-life with maxed-out TBW (Terabytes Written) and life percentage used.
* **Pattern 4 (Controller/Firmware Failure):** Controller instability characterized by massive spikes in read/write error rates without physical media errors.
* **Pattern 5 (Early-Life Defect):** Rapid error accumulation in early usage (<3000 hours), indicating manufacturing flaws.

*(Note: Thermal Failures and Power-Related Failures are supported by the architecture but were not present in the provided synthetic dataset.)*

## 💻 Installation & Execution

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/nvme-health-core.git](https://github.com/your-username/nvme-health-core.git)
cd nvme-health-core
