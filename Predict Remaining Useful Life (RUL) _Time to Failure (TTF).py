import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

 # --- Load Dataset ---
df = pd.read_csv("EV_Predictive_Maintenance.csv")

# Reduce dataset size for testing (optional)
df = df.sample(n=10000, random_state=42)

# --- Rename for clarity ---
df.rename(columns={
    "SoC": "State_of_Charge",
    "SoH": "State_of_Health",
    "RUL": "Remaining_Useful_Life",
    "TTF": "Time_to_Failure",
    "Route_Roughness": "Road_Profile_Irregularity"
}, inplace=True)

# Encode string columns - Transform numarical data to catorgorical data
le_failure = LabelEncoder()
df['Failure_Probability'] = le_failure.fit_transform(df['Failure_Probability'])

le_maint = LabelEncoder()
df['Maintenance_Type'] = le_maint.fit_transform(df['Maintenance_Type'])


#--- Categorical Decoding (or Inverse Label Encoding) ---Encode categorical columns
#convert integer-coded categorical values into human-readable string labels.
#df['Failure_Probability'] = df['Failure_Probability'].astype(int).map({0: 'No', 1: 'Yes'})
#Replaces those integers with meaningful labels
#df['Maintenance_Type'] = df['Maintenance_Type'].astype(int).map({ 0: 'Normal', 1: 'Preventive',
                                                                  #2: 'Corrective',  3: 'Predictive'})
# --- Feature Engineering ---
df['Battery_Power'] = df['Battery_Voltage'] * df['Battery_Current']
df['ΔSoC'] = df['State_of_Charge'].diff().fillna(0)
df['ΔVoltage'] = df['Battery_Voltage'].diff().fillna(0)

# Interactions
df['Current*Temp'] = df['Battery_Current'] * df['Battery_Temperature']
df['Temp_Diff'] = df['Battery_Temperature'] - df['Ambient_Temperature']

# --- Drop rows with any missing values ---
df.dropna(inplace=True)

# --- Define features and target (TTF or RUL) ---
target_col = 'Time_to_Failure'  # or use 'Remaining_Useful_Life'
features = df.drop(columns=[target_col])

X = features.select_dtypes(include=[np.number])
y = df[target_col]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Deep Learning Model ---linear stack of layers                   #ReLU (Rectified Linear Unit)
model = Sequential([Input(shape=(X_train_scaled.shape[1],)),Dense(128, activation='relu'),BatchNormalization(),
                    Dropout(0.3),Dense(64, activation='relu'),BatchNormalization(),Dropout(0.3),
                    Dense(32, activation='relu'),Dense(1)  ])

#Adam (Adaptive Moment Estimation): Mean Squared Error (MSE):Mean Absolute Error (MAE)
model.compile(optimizer=Adam(0.01), loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --- Train Model ---
history = model.fit(X_train_scaled, y_train,validation_split=0.2,epochs=50,batch_size=32,
                    callbacks=[early_stop],verbose=1)

# --- Evaluate Model ---
y_pred = model.predict(X_test_scaled).flatten()

def evaluate(y_true, y_pred):
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.4f}")

evaluate(y_test, y_pred)

# --- Visualizations ---
# Loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Val Loss")
plt.legend()
plt.title("Loss over Epochs")

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label="Train MAE")
plt.plot(history.history['val_mae'], label="Val MAE")
plt.legend()
plt.title("MAE over Epochs")
plt.tight_layout()
plt.show()

# --- Residual Plot ---
plt.figure(figsize=(8, 5))
sns.histplot(y_test - y_pred, bins=50, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.show()