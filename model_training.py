import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

file_path = r'NVMe_Drive_Failure_Dataset.csv'
df = pd.read_csv(file_path)

features = ['Temperature_C', 'Media_Errors', 'Unsafe_Shutdowns', 
            'Read_Error_Rate', 'Write_Error_Rate', 'Percent_Life_Used']

X = df[features]
y = df['Failure_Mode']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Applying SMOTE to balance the dataset...\n")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print("Training the Machine Learning Model... Please wait...\n")
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train_balanced, y_train_balanced)

predictions = model.predict(X_test)
print("--- MODEL ACCURACY & REPORT CARD ---")
print(f"Overall Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, 'nvme_model.pkl')
print("\nSUCCESS: The ML Model has been saved as 'nvme_model.pkl'!")
