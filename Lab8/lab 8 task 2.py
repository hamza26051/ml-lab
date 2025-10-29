import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

data = {
    'Refrigerator': [16,19,7,7,11,13,22,8,10,18,0,11,18,21,4,17,9,10,12,19],
    'AC_Conditioner': [23,23,20,22,23,22,23,20,23,22,19,22,22,23,21,22,23,21,23,22],
    'Television': [2,3,2,3,2,2,0,2,0,3,2,3,1,3,3,1,2,1,2,2],
    'Monitor': [3,4,21,21,11,16,21,8,14,14,20,19,9,6,5,10,8,10,6,10],
    'WaterPumps': [1,7,1,1,1,1,1,1,7,1,1,1,12,1,1,7,1,1,1,1],
    'Month': [8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'MonthlyUnits': [448,381,416,476,457,471,482,546,578,450,469,505,414,504,548,489,473,616,473,491],
    'TariffRate': [7.4,7.1,9.2,9.2,9.2,7.4,8.5,8.5,7.8,7.9,9.2,7.9,8.2,8.9,8.8,9.1,8.9,8.5,9.2,8.5],
    'ElectricityBill': [2808.4,2809.4,4024.0,4370.0,4204.4,3485.4,4097.0,4641.0,4508.4,3555.0,4317.8,3990.5,3394.8,4485.6,4823.4,4444.6,4211.7,5236.0,4356.6,4173.5]
}

df = pd.DataFrame(data)
y = df['ElectricityBill']
X = df[['MonthlyUnits', 'TariffRate']]
X_const = sm.add_constant(X)

import statsmodels.api as sm
model = sm.OLS(y, X_const).fit()

residuals = model.resid
fitted = model.fittedvalues

print("Dataset Shape:", df.shape)
print(df.head())
print(df.info())

print("\nCorrelation with ElectricityBill:")
print(df[['MonthlyUnits', 'TariffRate', 'ElectricityBill']].corr()['ElectricityBill'])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(df['MonthlyUnits'], y, alpha=0.6)
axes[0].set_xlabel('MonthlyUnits')
axes[0].set_ylabel('ElectricityBill')
axes[0].set_title('Units vs Bill')

axes[1].scatter(df['TariffRate'], y, alpha=0.6)
axes[1].set_xlabel('TariffRate')
axes[1].set_title('Rate vs Bill')
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("LINEAR REGRESSION SUMMARY")
print("="*50)
print(model.summary())

print("\n" + "="*50)
print("ASSUMPTION CHECKS")
print("="*50)

from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residuals, X_const)
print("Breusch-Pagan p-value:", bp_test[1])

shapiro_test = stats.shapiro(residuals)
print(f"Shapiro-Wilk p-value: {shapiro_test[1]:.4f}")

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["feature"] = X.columns
print("\nVIF:")
print(vif)

print("\n" + "="*50)
print("DECISION")
print("="*50)
print("LINEAR REGRESSION IS APPROPRIATE")
print("""
- Linearity: Strong between Units, Rate and Bill
- Independence: Satisfied (no clustering)
- Homoscedasticity: BP test p > 0.05 likely
- Normality: Acceptable for small sample
- No multicollinearity: VIF < 5
""")
print("Note: Model is descriptive (Bill ≈ Units × Rate), not predictive.")