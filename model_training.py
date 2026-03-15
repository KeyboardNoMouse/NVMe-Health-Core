import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. Load the data
file_path = r'NVMe_Drive_Failure_Dataset.csv'
df = pd.read_csv(file_path)

# 2. Select features and target
features = ['Temperature_C', 'Media_Errors', 'Unsafe_Shutdowns', 
            'Read_Error_Rate', 'Write_Error_Rate', 'Percent_Life_Used']

X = df[features]
y = df['Failure_Mode']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. APPLY SMOTE: Synthesize new failing drive data so the model can actually learn!
print("Applying SMOTE to balance the dataset... This is the secret sauce!\n")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# 5. Train the Model on the newly balanced data
print("Training the Machine Learning Model... Please wait...\n")
# Notice we removed the class_weight, SMOTE handles it now!
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train_balanced, y_train_balanced)

# 6. Test the model
predictions = model.predict(X_test)

# 7. Print the Report Card
print("--- MODEL ACCURACY & REPORT CARD ---")
print(f"Overall Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
print(classification_report(y_test, predictions, zero_division=0))

import joblib

# Save the trained model to a file so the website can use it!
joblib.dump(model, 'nvme_model.pkl')
print("\nSUCCESS: The ML Model has been saved as 'nvme_model.pkl'!")