import pandas as pd
import numpy as np
"""Initial Exploration Code"""

"""Load Dataset"""
df= pd.read_csv("EV_Predictive_Maintenance.csv")
print(df.head())
print(df.tail())

"""Show basic info """
print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())

"""Exploratory Data Analysis (EDA)"""

"""finding total null values in the dataset"""
finding_null = df.isna().sum().sum()
print(finding_null)

"""to find total duplicates"""
duplicate_rows = df[df.duplicated()]
print("Duplicate rows: ", duplicate_rows.shape)
numarical_data = [x for x in df.columns if df[x].dtype=="O"] #list comprehension
print(numarical_data)
finding_null_values_numarical= df[numarical_data].isnull().sum()
print(finding_null_values_numarical)
#
# """df.T represent transposes here Rows become columns and Columns become rows """
df= pd.read_csv("EV_Predictive_Maintenance.csv")
dup_col_mask = df.T.duplicated()
duplicate_columns = df.columns[dup_col_mask]
print("Duplicate column names:", duplicate_columns.tolist())

"""Rename the columns"""
df= df.rename(columns={"SoC":"State of Charge","SoH":"State of Health","RUL":"Remaining Useful Life","TTF":"Time to Failure",
                       "Route_Roughness":"Road Profile Irregularity"})
print(df.columns)

"""Failure_Probability (Map 0 to 'No' and 1 to 'Yes')"""
df['Failure_Probability'] = df['Failure_Probability'].astype(int).map({0: 'No', 1: 'Yes'})
print(df['Failure_Probability'].unique())
print(df['Failure_Probability'].dtype)

"""Maintenance_Type (Map 0 to 'Normal' and 1 to 'Preventive' and 2 to 'Corrective' and 3 to 'Predictive')"""
df['Maintenance_Type'] = df['Maintenance_Type'].astype(int).map({0: 'Normal', 1: 'Preventive', 2: 'Corrective', 3: 'Predictive'})
print(df['Maintenance_Type'].unique())
print(df['Maintenance_Type'].dtype)
pd.set_option('display.max_columns', None)  # Show all columns
print(df.head())



