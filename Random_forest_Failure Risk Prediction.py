import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import numpy as np

# Load dataset
df = pd.read_csv("EV_Predictive_Maintenance.csv")

# Reduce dataset size for testing (optional)
df = df.sample(n=20000, random_state=42)

# ============================================
# PART 1: MULTI-CLASS CLASSIFICATION
# ============================================

print("\n" + "="*50)
print("MULTI-CLASS CLASSIFICATION (Failure Type Analysis)")
print("="*50)

# Define failure label
def classify_failure(row):
    if row['Failure_Probability'] == 0 or row['Maintenance_Type'] == 0:
        return 'No Failure'
    elif row['Maintenance_Type'] == 1:
        return 'Battery Failure'
    elif row['Maintenance_Type'] == 2:
        return 'Motor Failure'
    elif row['Maintenance_Type'] == 3:
        return 'Brake Failure'
    else:
        return 'Unknown'

df['Failure_Label'] = df.apply(classify_failure, axis=1)

# Select features and target
X_multi = df.drop(columns=['Timestamp', 'Failure_Probability', 'Maintenance_Type', 'Failure_Label'])
y_multi = df['Failure_Label']

# Encode target labels - convert categorical target labels (y_multi) into numerical values
label_encoder = LabelEncoder()
y_multi_encoded = label_encoder.fit_transform(y_multi)

# Scale features -#removing the mean and scaling to unit variance
scaler = StandardScaler()
X_multi_scaled = scaler.fit_transform(X_multi) #calculates the mean and std dev

# Show class distribution before balancing
print("\nClass distribution before balancing:")
print(Counter(y_multi_encoded))

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_multi_resampled, y_multi_resampled = smote.fit_resample(X_multi_scaled, y_multi_encoded) #applies the oversampling to create balanced classes

# Show class distribution after balancing
print("\nClass distribution after SMOTE balancing:")
print(Counter(y_multi_resampled))

# Train/test split
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi_resampled, y_multi_resampled, test_size=0.2, random_state=42, stratify=y_multi_resampled)

# Train Random Forest model
rf_multi = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight='balanced', n_jobs=-1, random_state=42)
rf_multi.fit(X_train_multi, y_train_multi)

# Predict and evaluate
y_pred_multi = rf_multi.predict(X_test_multi)

print("\nClassification Report for Failure Type Prediction:")
print(classification_report(y_test_multi, y_pred_multi, target_names=label_encoder.classes_))

print(f"Accuracy: {accuracy_score(y_test_multi, y_pred_multi) * 100:.2f}%")

# Confusion Matrix
plt.figure(figsize=(8,6))
cm_multi = confusion_matrix(y_test_multi, y_pred_multi)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Failure Type Prediction")
plt.show()

# Feature Importance  - how much each input feature (column) in your dataset affects the model's predictions.
importances = rf_multi.feature_importances_
indices = np.argsort(importances)[::-1]
features_sorted = X_multi.columns[indices]

plt.figure(figsize=(10,6))
sns.barplot(x=importances[indices], y=features_sorted)
plt.title("Feature Importance - Multi-Class Failure Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Visualize one tree
plt.figure(figsize=(12,10))
plot_tree(rf_multi.estimators_[0], feature_names=X_multi.columns,
          class_names=label_encoder.classes_, filled=True,rounded=True, max_depth=3, proportion=True)
plt.title("Random Forest Tree (Multi-Class Failure)")
plt.show()

# ============================================
# PART 2: BINARY CLASSIFICATION
# ============================================

print("\n" + "="*50)
print("BINARY CLASSIFICATION (Failure/No Failure)")
print("="*50)

# Prepare binary labels
y_binary = df['Failure_Probability'].apply(lambda x: 1 if x > 0 else 0)
X_binary = df.drop(columns=['Timestamp', 'Failure_Probability', 'Maintenance_Type', 'Failure_Label'])

# Scale using same scaler
X_binary_scaled = scaler.transform(X_binary)

# Class distribution before balancing
print("\nClass distribution before balancing:")
print(Counter(y_binary))

# Apply SMOTE
X_binary_resampled, y_binary_resampled = smote.fit_resample(X_binary_scaled, y_binary)

# Train/test split
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    X_binary_resampled, y_binary_resampled, test_size=0.2, random_state=42, stratify=y_binary_resampled)

# Train Random Forest model
rf_binary = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight='balanced', random_state=42)
rf_binary.fit(X_train_binary, y_train_binary)

# Predict and evaluate
y_pred_binary = rf_binary.predict(X_test_binary)

print("\nClassification Report for Binary Failure Prediction:")
print(classification_report(y_test_binary, y_pred_binary, target_names=['No Failure', 'Failure']))
print(f"Accuracy: {accuracy_score(y_test_binary, y_pred_binary) * 100:.2f}%")

# Confusion Matrix
plt.figure(figsize=(6,5))
cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Greens',xticklabels=['No Failure', 'Failure'],
            yticklabels=['No Failure', 'Failure'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Binary Failure Prediction")
plt.show()

# Feature Importance
importances_bin = rf_binary.feature_importances_
indices_bin = np.argsort(importances_bin)[::-1]

plt.figure(figsize=(10,6))
sns.barplot(x=importances_bin[indices_bin], y=X_binary.columns[indices_bin])
plt.title("Feature Importance - Binary Failure Prediction")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Visualize one tree
plt.figure(figsize=(12,8))
plot_tree(rf_binary.estimators_[0], feature_names=X_binary.columns,
          class_names=['No Failure', 'Failure'], filled=True,
          rounded=True, max_depth=3, proportion=True)
plt.title("Random Forest Tree (Binary Failure)")
plt.show()