import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings

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
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])

print("\nMelatih Model AI...")
rf_pipeline.fit(X_train, y_train)

akurasi = rf_pipeline.score(X_test, y_test)
print(f"Training Selesai! Akurasi Model: {akurasi * 100:.2f}%")

print("\nMenyimpan model...")
model_path = 'models/model_rekomendasi.pkl'
joblib.dump(rf_pipeline, model_path)
print(f"Model berhasil disimpan di: {model_path}")