
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

import joblib
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

df = pd.read_csv('cars.csv')
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")
print(df.head())


print("\nData Types and Non-Null Counts:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe(include='all').T)

print("\nMissing Values (%):")
missing = df.isnull().mean().sort_values(ascending=False)
print(missing[missing > 0])

plt.figure()
sns.histplot(df['price'], kde=True, bins=50, color='skyblue')
plt.title('Distribution of Car Prices')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()
numeric_cols = df.select_dtypes(include=np.number).columns
corr = df[numeric_cols].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, linewidths=0.5)
plt.title('Correlation Matrix (Numeric Features)')
plt.show()

cat_cols = ['symboling', 'CarName', 'fueltype', 'aspiration', 'doornumber',
            'carbody', 'drivewheel', 'enginelocation', 'enginetype',
            'cylindernumber', 'fuelsystem']

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.ravel()
for i, col in enumerate(cat_cols):
    sns.boxplot(x=col, y='price', data=df, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} vs Price')
    axes[i].tick_params(axis='x', rotation=45)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

X = df.drop(columns=['price'])
y = df['price']
numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"Numeric Features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical Features ({len(categorical_features)}): {categorical_features}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")


print("\n" + "="*50)
print("TRAINING LINEAR REGRESSION")
print("="*50)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X_train, y_train)
print("Model training completed.")

print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:5} → MAE: {mae:8.0f} | RMSE: {rmse:8.0f} | R²: {r2:.4f}")
    return mae, rmse, r2

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mae, train_rmse, train_r2 = evaluate(y_train, y_train_pred, "TRAIN")
test_mae,  test_rmse,  test_r2  = evaluate(y_test,  y_test_pred,  "TEST")

cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"\n5-Fold CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\n" + "="*50)
print("PLOTTING RESULTS")
print("="*50)

plt.figure()
plt.scatter(y_train, y_train_pred, alpha=0.6, label='Train', color='steelblue')
plt.scatter(y_test, y_test_pred, alpha=0.6, label='Test', color='crimson')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('True Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression: True vs Predicted Price')
plt.legend()
plt.grid(True)
plt.show()

residuals_train = y_train - y_train_pred
residuals_test = y_test - y_test_pred

plt.figure()
sns.histplot(residuals_train, kde=True, label='Train', alpha=0.7, color='blue')
sns.histplot(residuals_test, kde=True, label='Test', alpha=0.7, color='red')
plt.title('Residuals Distribution')
plt.xlabel('Residual (True - Predicted)')
plt.legend()
plt.show()

ohe = model.named_steps['preprocessor'].named_transformers_['cat']['onehot']
cat_feature_names = ohe.get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_features, cat_feature_names])

coef = model.named_steps['regressor'].coef_
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})
coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
top10 = coef_df.sort_values('AbsCoef', ascending=False).head(10)

plt.figure()
sns.barplot(x='Coefficient', y='Feature', data=top10, palette='viridis')
plt.xlabel('Coefficient Value')
plt.show()

joblib.dump(model, 'car_price_linear_regression_model.pkl')
print(f"Train R² : {train_r2:.4f}")
print(f"Test  R² : {test_r2:.4f}")
print(f"CV R²    : {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(top10[['Feature', 'Coefficient']].head(3).to_string(index=False))
