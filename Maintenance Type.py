import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Load dataset
df = pd.read_csv("EV_Predictive_Maintenance.csv")
df = df.sample(n=20000, random_state=42)  # Reduce for speed

# ============================================
# MULTI-CLASS CLASSIFICATION: MAINTENANCE TYPE
# ============================================

print("\n" + "="*50)
print("MAINTENANCE TYPE PREDICTION - XGBoost")
print("="*50)

# Define target and features
X = df.drop(columns=['Timestamp', 'Failure_Probability', 'Maintenance_Type'])
y = df['Maintenance_Type']

# Class distribution before encoding
label_names = {
    0: 'No Maintenance',
    1: 'Battery',
    2: 'Motor',
    3: 'Brake'
}
print("\nMaintenance Type Distribution:")
print(Counter(y.map(label_names)))

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print("\nClass distribution after SMOTE balancing:")
print(Counter(y_resampled.map(label_names)))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(objective='multi:softprob',num_class=4,learning_rate=0.1,n_estimators=200,max_depth=6,
                              subsample=0.8,colsample_bytree=0.8,eval_metric='mlogloss',use_label_encoder=False,
                              random_state=42)

xgb_model.fit(X_train, y_train)

# Predictions
y_pred = xgb_model.predict(X_test)

# Evaluation
print("\nClassification Report (Maintenance Type):")
print(classification_report(y_test, y_pred, target_names=[label_names[i] for i in sorted(label_names)]))
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Confusion matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=[label_names[i] for i in sorted(label_names)],
            yticklabels=[label_names[i] for i in sorted(label_names)])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost Maintenance Type")
plt.tight_layout()
plt.show()

# Feature importance
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X.columns[indices]

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.title("Feature Importance - XGBoost Maintenance Type")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
