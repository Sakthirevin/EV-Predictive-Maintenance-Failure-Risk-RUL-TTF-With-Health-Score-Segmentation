import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("EV_Predictive_Maintenance.csv")

# Drop non-numeric & target columns for scaling and modeling
X = df.drop(columns=['Timestamp', 'Failure_Probability', 'Maintenance_Type'], errors='ignore')

# Check infinities
print("Infs:\n", np.isinf(X).sum())

# Normalize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.select_dtypes(include=[np.number]))

# Correlation matrix (Spearman) for numeric columns
df_numeric = df.select_dtypes(include=[np.number])


# --- Remove outliers using IQR ---
def remove_outliers_iqr(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = df_clean[col].where((df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound), np.nan)
    return df_clean

df_clean = remove_outliers_iqr(df_numeric)

# Drop rows with missing values after IQR cleaning
df_clean = df_clean.dropna()

# Log-transform and scale cleaned data
df_log_scaled = np.log1p(df_clean)
df_log_scaled = pd.DataFrame(StandardScaler().fit_transform(df_log_scaled), columns=df_log_scaled.columns)

# --- Boxplot of log-scaled outlier-free data ---
plt.figure(figsize=(14, 10))
sns.set(style="whitegrid")
sns.boxplot(data=df_log_scaled, orient="h", palette="Spectral", fliersize=0.5, linewidth=1)
plt.title("Outlier-Free Log-Scaled Boxplot of EV Parameters", fontsize=16, weight='bold')
plt.xlabel("Log Scaled Value", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Motor Performance Pairplot ---
sns.set(style="ticks", font_scale=1)

# Subset dataframe for these variables
cols = ['Motor_RPM', 'Motor_Temperature', 'Motor_Torque', 'Motor_Vibration']

sns.pairplot(df[['Motor_Torque','Motor_RPM','Motor_Temperature','Motor_Vibration']].sample(1000),
             height=2.5, aspect=1, kind="scatter", diag_kind="kde", corner=True)
plt.show()

# --- Correlation Heatmap ---
# Select specific columns for the heatmap
selected_columns = df[['SoH', 'Battery_Temperature', 'Motor_Temperature', 'Tire_Temperature', 'Ambient_Temperature']]
# Compute the correlation matrix
correlation_matrix = selected_columns.corr()
#Figuring the HeatMap
plt.figure(figsize=(10, 6))  # Set the figure size
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title("Heatmap of Thermal Metrics")
plt.show()

# --- Isolation Forest Anomaly Detection ---
iso_forest = IsolationForest(contamination=0.01, random_state=42)
df['Anomaly_Score'] = iso_forest.fit_predict(X_scaled)

df['Anomaly'] = df['Anomaly_Score'].apply(lambda x: 1 if x == -1 else 0)
anomaly_count = df['Anomaly'].value_counts()

plt.figure(figsize=(6, 4))
anomaly_count.plot(kind='bar', color=['green', 'red'])
plt.xticks([0, 1], ['Normal', 'Anomaly'], rotation=0)
plt.ylabel("Count")
plt.title("Anomaly Detection with Isolation Forest")
plt.tight_layout()
plt.show()

# Preview some anomalies
print("Anomalous Samples Preview:")
print(df[df['Anomaly'] == 1].head())
