import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder


columns = [
    "ID", "Diagnosis", "radius1", "texture1", "perimeter1", "area1", "smoothness1", "compactness1", 
    "concavity1", "concave_points1", "symmetry1", "fractal_dimension1", "radius2", "texture2", 
    "perimeter2", "area2", "smoothness2", "compactness2", "concavity2", "concave_points2", 
    "symmetry2", "fractal_dimension2", "radius3", "texture3", "perimeter3", "area3", 
    "smoothness3", "compactness3", "concavity3", "concave_points3", "symmetry3", 
    "fractal_dimension3"
]

data = pd.read_csv('breastcancer.data', header=None, names=columns)



data['Diagnosis'] = LabelEncoder().fit_transform(data['Diagnosis'])
data = data.drop(columns=['ID']) 

X = data.drop(columns=['Diagnosis'])
y = data['Diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


def calculate_f_scores(X, y):
    f_scores = []
    
    for i in range(X.shape[1]):
        feature = X[:, i]
        mean_pos = np.mean(feature[y == 1])
        mean_neg = np.mean(feature[y == 0])
        mean_total = np.mean(feature)
        numerator = (mean_pos - mean_total) ** 2 + (mean_neg - mean_total) ** 2
        denom_pos = np.var(feature[y == 1])
        denom_neg = np.var(feature[y == 0])
        denominator = (1 / len(feature[y == 1]) * denom_pos) + (1 / len(feature[y == 0]) * denom_neg)
        f_scores.append(numerator / denominator)
        
    return np.array(f_scores)

f_scores = calculate_f_scores(X_scaled, y)
features_sorted = np.argsort(f_scores)[::-1]

feature_subsets = [features_sorted[:i] for i in range(1, 10)]


def evaluate_svm(X_train, X_test, y_train, y_test, features):
    X_train_sub = X_train[:, features]
    X_test_sub = X_test[:, features]
    
    model = SVC(kernel='rbf', C=1, gamma=0.1) 
    model.fit(X_train_sub, y_train)
    y_pred = model.predict(X_test_sub)
    
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) 
    
    return accuracy, sensitivity, specificity, cm


def evaluate(X, y, splits, feature_subsets):
    results = []
    for split_name, test_size in splits.items():
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

        for i, features in enumerate(feature_subsets):
            accuracy, sensitivity, specificity, cm = evaluate_svm(X_train, X_test, y_train, y_test, features)
            results.append({
                'Split': split_name,
                'Model': f"Model #{i+1}",
                'Features': len(features),
                'Accuracy': accuracy,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'Confusion Matrix': cm
            })
    return results


splits = {'50-50': 0.5, '70-30': 0.3, '80-20': 0.2}

results = evaluate(X_scaled, y, splits, feature_subsets)

results_df = pd.DataFrame(results)


table_4 = results_df.pivot_table(index=['Model'], columns='Split', values='Accuracy', aggfunc='mean')
print("\nTable 4:")
print(table_4)

model_5_results = results_df[results_df['Model'] == 'Model #5']

print("\nTable 6 and 7:")
for idx, res in model_5_results.iterrows():
    print(f"\n{res['Split']}:")
    print(f"Sensitivity: {res['Sensitivity']}, Specificity: {res['Specificity']}")
    print(f"Confusion Matrix:\n{res['Confusion Matrix']}")