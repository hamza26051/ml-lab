import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

columns = [
    "ID", "Diagnosis", "radius1", "texture1", "perimeter1", "area1", "smoothness1", "compactness1", 
    "concavity1", "concave_points1", "symmetry1", "fractal_dimension1", "radius2", "texture2", 
    "perimeter2", "area2", "smoothness2", "compactness2", "concavity2", "concave_points2", 
    "symmetry2", "fractal_dimension2", "radius3", "texture3", "perimeter3", "area3", 
    "smoothness3", "compactness3", "concavity3", "concave_points3", "symmetry3", 
    "fractal_dimension3"
]

data = pd.read_csv('breastcancer.data', header=None, names=columns)

data.shape
data.info()
data.head()


missing_values = data.isnull().sum()

data['Diagnosis'] = LabelEncoder().fit_transform(data['Diagnosis'])

data = data.drop(columns=['ID'])

X = data.drop(columns=['Diagnosis'])
y = data['Diagnosis']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled.shape, y.shape



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, stratify=y, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='rbf', probability=True, random_state=42)
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)
y_pred_svm = svm.predict(X_test)

def calculate_metrics(y_true, y_pred):
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    gmean = (sensitivity * specificity) ** 0.5
    fdr = fp / (fp + tp) 
    forate = fn / (fn + tn) 
    mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    
    return {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'G-Mean': gmean,
        'FDR': fdr,
        'FOR': forate,
        'MCC': mcc
    }



metrics_knn = calculate_metrics(y_test, y_pred_knn)
metrics_svm = calculate_metrics(y_test, y_pred_svm)


metrics_results = pd.DataFrame([metrics_knn, metrics_svm], index=['KNN', 'SVM']).T

print("Metrics Results:")
print(metrics_results)



fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_knn = confusion_matrix(y_test, y_pred_knn)
disp_knn = ConfusionMatrixDisplay(confusion_matrix=cm_knn)
disp_knn.plot(cmap=plt.cm.Blues, ax=axes[0], values_format='d')
axes[0].set_title('KNN Confusion Matrix')


cm_svm = confusion_matrix(y_test, y_pred_svm)
disp_svm = ConfusionMatrixDisplay(confusion_matrix=cm_svm)
disp_svm.plot(cmap=plt.cm.Blues, ax=axes[1], values_format='d')
axes[1].set_title('SVM Confusion Matrix')

plt.tight_layout()
plt.show()

metrics_results.plot(kind='bar', figsize=(10, 6), colormap='viridis', edgecolor='black')
plt.title('Performance Metrics for KNN and SVM')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.legend(title='Classifier')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()