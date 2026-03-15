import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
file_path = r'NVMe_Drive_Failure_Dataset.csv' 
df = pd.read_csv(file_path)

# 2. Count occurrences of each Failure Mode
print("--- Count of Drives per Failure Mode ---")
print(df['Failure_Mode'].value_counts().sort_index())
print("\n" + "="*50 + "\n")

# 3. Analyze hardware metrics across the failure patterns
metrics_to_analyze = [
    'Temperature_C', 'Media_Errors', 'Unsafe_Shutdowns', 
    'Read_Error_Rate', 'Write_Error_Rate', 'Percent_Life_Used'
]

pattern_analysis = df.groupby('Failure_Mode')[metrics_to_analyze].mean()

print("--- Average Hardware Metrics per Failure Pattern ---")
print(pattern_analysis.round(2))
print("\n" + "="*50 + "\n")

# 4. Generate Visualization for the Lenovo Presentation
plt.figure(figsize=(14, 8))
plt.suptitle('Hardware Metrics Across Failure Patterns (0 = Healthy)', fontsize=16, fontweight='bold')

for i, metric in enumerate(metrics_to_analyze, 1):
    plt.subplot(2, 3, i)
    sns.barplot(x=pattern_analysis.index, y=pattern_analysis[metric], palette="coolwarm", hue=pattern_analysis.index, legend=False)
    plt.title(f'Average {metric}')
    plt.xlabel('Failure Pattern')
    plt.ylabel(metric)

plt.tight_layout()
plt.savefig('Failure_Patterns_Dashboard.png')
print("SUCCESS: 'Failure_Patterns_Dashboard.png' generated for your slides.")
