# 🧠 Loan Default Prediction using Logistic Regression

This project predicts whether a person is likely to default on a loan based on their income, loan amount, and employment status. It uses logistic regression and visualizes the prediction on a sigmoid curve.

# 📘 Key Terms Explained
# ✅ Repay
Meaning: To pay back the loan amount (plus interest) to the bank on time.
Example:
Ravi takes a ₹1,00,000 loan from the bank. He pays ₹5,000 every month for 2 years. He never misses a payment.
👉 Ravi repaid the loan

# ⚠️ Default
Meaning: To fail to pay back the loan as agreed — either by missing payments or stopping altogether.
Example:
Priya takes a ₹2,00,000 loan but loses her job. She stops paying after 3 months.
👉 Priya defaulted on the loan.

## 📦 Dataset
- File: `loan_default_prediction.csv`
- Columns: `income`, `loan_amount`, `employment_status`, `default`

## 🚀 How to Run

1. Make sure `loan_default_prediction.csv` is in the same folder as the Python script.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

# Output or result--

✅ Model Accuracy: 73.67 %

📊 Classification Report:
               precision    recall  f1-score   support

           0       0.81      0.72      0.76       174
           1       0.66      0.76      0.71       126

    accuracy                           0.74       300
   macro avg       0.73      0.74      0.73       300
weighted avg       0.75      0.74      0.74       300


🔍 Enter details to predict loan default:
Enter monthly income (e.g., 40000): 40000
Enter loan amount (e.g., 150000): 150000
Enter employment status (Employed/Unemployed): Employed

📈 Prediction Result:
✅ Likely to REPAY the loan.
Probability of default: 8.8 %