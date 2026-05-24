import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV 
import joblib
import warnings
import json 
import os 

# Kode untuk MLflow dan DagsHub
import dagshub
import mlflow
from dotenv import load_dotenv

load_dotenv() # Membaca token dari file .env

# Mengaktifkan koneksi ke DagsHub
dagshub.init(
    repo_owner=os.getenv("MLFLOW_TRACKING_USERNAME"), 
    repo_name="sistem_rekomendasi_lari_PDM", 
    mlflow=True
)

warnings.filterwarnings('ignore')

print("Start Training...")
# Membaca file
try:
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')
except FileNotFoundError:
    print("ERROR:File train.csv atau test.csv tidak ditemukan!")
    print("Jalankan 'python src/data_preparation.py' terlebih dahulu.")
    exit()

# Fitur (X) dan Target (y)
fitur = ['Athlete age', 'Athlete average speed', 'Distance_num', 'Athlete gender']
X_train, y_train = train_df[fitur], train_df['Level']
X_test, y_test = test_df[fitur], test_df['Level']

print("\nMembangun MLOps Pipeline...")
numeric_features = ['Athlete age', 'Athlete average speed', 'Distance_num']
categorical_features = ['Athlete gender']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Model Random Forest untuk klasifikasi level
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Mencari Model Terbaik (Hyperparameter Tuning)
print("\nMelatih dan Membandingkan Beberapa Variasi Model...")

param_grid = {
    'classifier__n_estimators': [50, 100, 150],
    'classifier__max_depth': [None, 10, 20]
}

# Membuat mesin untuk mencari model terbaik
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, n_jobs=-1, verbose=0) # verbose diubah ke 0 agar print manual kita yang muncul
grid_search.fit(X_train, y_train)

# Menampilkan semua model yang diuji
print("\nHasil pengujian beberapa model (Hyperparameter Tuning):")
print("-" * 60)
hasil_uji = grid_search.cv_results_
for mean_score, params in zip(hasil_uji['mean_test_score'], hasil_uji['params']):
    pohon = params['classifier__n_estimators']
    kedalaman = params['classifier__max_depth']
    akurasi_persen = mean_score * 100
    print(f"Model dengan {pohon} Pohon & Kedalaman {kedalaman} \t-> Akurasi Validasi: {akurasi_persen:.2f}%")
print("-" * 60)

# Mengambil model yang paling terbaik
best_pipeline = grid_search.best_estimator_
best_params = grid_search.best_params_

print(f"\nBest model ditemukan:")
print(f"Versi terbaik adalah: {best_params['classifier__n_estimators']} Pohon dengan Kedalaman {best_params['classifier__max_depth']}")

# Menggunakan model terbaik untuk menghitung akurasi di data test
akurasi = best_pipeline.score(X_test, y_test)
print(f"Training Selesai! Akurasi Model Terbaik pada Data Testing: {akurasi * 100:.2f}%")

# --- KODE BARU: MENGUNGGAH KE MLFLOW CLOUD ---
print("\nMengunggah log dan model ke MLflow DagsHub...")
mlflow.set_experiment("Eksperimen_Rekomendasi_Lari")

with mlflow.start_run():
    # Catat parameter dan metrik akurasi ke Cloud
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", akurasi)
    
    # Daftarkan model terbaik ke Registry Cloud dengan nama "Model_Rekomendasi_Lari"
    mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        artifact_path="model",
        registered_model_name="Model_Rekomendasi_Lari"
    )
print("Model berhasil diunggah dan diregistrasi di MLflow!")

# Log Experiment (Lokal)
print("\nMencatat Log Experiment Lokal...")
experiment_log = {
    "model_type": "Random Forest Pipeline",
    "best_parameters": best_params,
    "accuracy": akurasi,
    "total_training_samples": len(train_df)
}

os.makedirs('reports', exist_ok=True)
with open('reports/metrics.json', 'w') as f:
    json.dump(experiment_log, f, indent=4)
print("Log Experiment berhasil disimpan di: reports/metrics.json")

print("\nMenyimpan model lokal...")
model_path = 'models/model_rekomendasi.pkl'
# Menyimpan best_pipeline
joblib.dump(best_pipeline, model_path)
print(f"Model lokal berhasil disimpan di: {model_path}")
print("Seluruh proses selesai!")