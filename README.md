# ⚡ NVMe Health Core: Predictive Telemetry Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

An AI-powered diagnostics engine and full-stack web interface designed to predict NVMe drive failures before they happen. By analyzing raw SMART telemetry data, this system isolates hardware degradation signatures and categorizes them into distinct failure patterns.

## 🎯 The Problem
Data centers and consumer PCs rely heavily on NVMe drives, but sudden hardware failures can lead to catastrophic data loss. While drives report SMART (Self-Monitoring, Analysis, and Reporting Technology) telemetry, manually deciphering this data is inefficient. This project automates that process using machine learning to detect anomalies based on temperature spikes, media errors, and controller wear.

## 🚀 Key Features
* **Machine Learning Engine:** Utilizes a Random Forest Classifier to identify complex, non-linear relationships in hardware telemetry.
* **Advanced Data Engineering:** Implements SMOTE (Synthetic Minority Over-sampling Technique) to solve severe class imbalance, allowing the model to detect extremely rare controller failures.
* **Full-Stack Dashboard:** A sleek, interactive web interface built with Streamlit, custom CSS glassmorphism, and Lottie animations. 
* **Real-Time Diagnostics:** Users can upload raw CSV server logs and instantly receive a parsed, color-coded health report for their entire drive fleet.

## 🧠 The Data Science Process
1. **Exploratory Data Analysis (EDA):** Analyzed 10,000 rows of drive data to identify three primary failure modes: End of Life (Wear-Out), Media Corruption, and Controller Errors.
2. **Feature Engineering:** Extracted key warning signs, determining that `Percent_Life_Used` and `Read_Error_Rate` were the highest predictors of imminent failure.
3. **Handling Class Imbalance:** The dataset contained 98% healthy drives and <2% failing drives. Standard accuracy metrics were misleading (the "Accuracy Trap"). We deployed SMOTE to synthesize minority class data, vastly improving the model's recall for rare corruption events.

## 💻 Installation & Usage

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/nvme-health-core.git](https://github.com/your-username/nvme-health-core.git)
cd nvme-health-core
