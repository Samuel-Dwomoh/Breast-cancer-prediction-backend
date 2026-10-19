import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

data = pd.read_csv('breast_cancer.csv')
data.head()
data.describe()
data.info()

X = data.drop(columns=['id', 'diagnosis'])
le= LabelEncoder()
y = le.fit_transform(data['diagnosis'])
print(data.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

preprocessing = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric)
    ]
)

model = Pipeline(
    steps=[
        ('preprocessing', preprocessing),
        ('classifier', LogisticRegression(C=0.1, solver='liblinear', max_iter=500))
    ]
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

grid_params = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'saga']}

grid_search = GridSearchCV(model, grid_params, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

joblib.dump(grid_search.best_estimator_, 'model.joblib')
print("Model saved successfully as model.joblib!")

y2_pred = grid_search.predict(X_test)
print(f'Accuracy score: {accuracy_score(y_test, y2_pred)*100:.2f}%')
print(f'Confusion matrix:\n{confusion_matrix(y_test, y2_pred)}')
print(f'Classification report:\n{classification_report(y_test, y2_pred, target_names=le.classes_)}')