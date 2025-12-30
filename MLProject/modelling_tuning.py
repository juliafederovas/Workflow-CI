import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os
import joblib

# 1. Konfigurasi DagsHub & MLflow (Headless Auth)
repo_owner = 'juliafederovas'
repo_name = 'Eksperimen_SML_Julia-Federova-Sitio'
token = os.getenv("MLFLOW_TRACKING_PASSWORD")

if token:
    os.environ['MLFLOW_TRACKING_USERNAME'] = token
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token
    mlflow.set_tracking_uri(f'https://dagshub.com/{repo_owner}/{repo_name}.mlflow')
    print("Autentikasi DagsHub via Token Berhasil.")
else:
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

# 2. Setup Path & Load Data
mlflow.set_experiment("Occupancy_Estimation_Baseline")
base_dir = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(base_dir, "occupancy_processed.csv") 
mlflow.autolog()

if not os.path.exists(path_data):
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {path_data}")

df = pd.read_csv(path_data)
X = df.drop(columns=['Room_Occupancy_Count'])
y = df['Room_Occupancy_Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Training & Hyperparameter Tuning

param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3)
     
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
     
if token:
    mlflow.sklearn.save_model(best_model, "mlruns/model")
 
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) 
    mlflow.log_params(grid_search.best_params_) 
    mlflow.log_metric("accuracy", acc)
 
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Julia')
    path_cm = os.path.join(base_dir, "training_confusion_matrix.png")
    plt.savefig(path_cm)
    mlflow.log_artifact(path_cm) 
    plt.close() 
 
    plt.figure(figsize=(8,6))
    feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importance - Julia')
    path_fi = os.path.join(base_dir, "feature_importance.png")
    plt.savefig(path_fi)
    mlflow.log_artifact(path_fi) 
    plt.close()
     
    path_model = os.path.join(base_dir, "model.pkl")
    joblib.dump(best_model, path_model)
    print(f"Training selesai. Akurasi: {acc}")
