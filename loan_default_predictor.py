import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os

# 🔹 Step 1: Load dataset (must be in same folder)
csv_file = 'loan_default_prediction.csv'
if not os.path.exists(csv_file):
    print(f"❌ File '{csv_file}' not found. Please place it in the same folder as this script.")
    exit()

df = pd.read_csv(csv_file)

# 🔹 Step 2: Preprocessing
df.dropna(inplace=True)
le = LabelEncoder()
df['employment_status'] = le.fit_transform(df['employment_status'])

# 🔹 Step 3: Features and target
X = df[['income', 'loan_amount', 'employment_status']]
y = df['default']

# 🔹 Step 4: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔹 Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 🔹 Step 6: Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# 🔹 Step 7: Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n✅ Model Accuracy:", round(accuracy * 100, 2), "%")
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

# 🔹 Step 8: User input
print("\n🔍 Enter details to predict loan default:")
income = float(input("Enter monthly income (e.g., 40000): "))
loan_amount = float(input("Enter loan amount (e.g., 150000): "))
employment = input("Enter employment status (Employed/Unemployed): ")

# 🔹 Step 9: Encode and scale input
employment_encoded = le.transform([employment])[0]
user_input = pd.DataFrame([[income, loan_amount, employment_encoded]], columns=['income', 'loan_amount', 'employment_status'])
user_scaled = scaler.transform(user_input)

# 🔹 Step 10: Predict
prediction = model.predict(user_scaled)[0]
probability = model.predict_proba(user_scaled)[0][1]

# 🔹 Step 11: Output result
print("\n📈 Prediction Result:")
if prediction == 1:
    print("⚠️ Likely to DEFAULT on loan.")
else:
    print("✅ Likely to REPAY the loan.")
print("Probability of default:", round(probability * 100, 2), "%")

# 🔹 Step 12: Sigmoid curve with prediction point
z = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-z))
user_z = np.dot(user_scaled, model.coef_.T) + model.intercept_

plt.figure(figsize=(8, 5))
plt.plot(z, sigmoid, color='orange', label='Sigmoid Curve')
plt.axhline(0.5, color='gray', linestyle='--', label='Decision Boundary')
plt.scatter(user_z, probability, color='red', label='Your Prediction')
plt.title("Sigmoid Function with Prediction Point")
plt.xlabel("Z = β₀ + β₁x₁ + β₂x₂")
plt.ylabel("Probability of Default")
plt.legend()
plt.grid(True)
plt.show()

