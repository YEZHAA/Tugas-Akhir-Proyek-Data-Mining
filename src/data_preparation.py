import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

print("Mulai.............")

# Data Acquisition(Pengumpulan data)
print("\nTahap data acquisition...")
file_path = 'data/raw/Data_running.csv'

try:
    df_raw = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')
    print(f"Data berhasil diakuisisi, Total baris awal: {df_raw.shape[0]}")
except FileNotFoundError:
    print(f"ERROR: File sumber {file_path} tidak ditemukan.")
    exit()

# Data Cleaning dan Feature Engineering
print("\nData cleaning dan Feature engineering...")
# Mengambil kolom yang ada di data mentah
kolom_mentah = [
    'Year of event', 
    'Athlete year of birth', 
    'Athlete average speed', 
    'Event distance/length', 
    'Athlete gender'
]
df_clean = df_raw[kolom_mentah].dropna().head(10000).copy()

# Menghitung Umur (Tahun Lomba - Tahun Lahir)
df_clean['Athlete age'] = (df_clean['Year of event'] - df_clean['Athlete year of birth']).astype(int)

# Memfilter umur 15-80 tahun
df_clean = df_clean[(df_clean['Athlete age'] >= 15) & (df_clean['Athlete age'] <= 80)]

# Standardisasi Kolom Gender agar bersih untuk AI
df_clean = df_clean[df_clean['Athlete gender'].isin(['M', 'F', 'm', 'f'])]
df_clean['Athlete gender'] = df_clean['Athlete gender'].str.upper()

# Mengubah tipe datta kecepatan jadi angka
df_clean['Athlete average speed'] = pd.to_numeric(df_clean['Athlete average speed'], errors='coerce')

# Mengekstrak angka jarak
df_clean['Distance_num'] = df_clean['Event distance/length'].str.extract(r'(\d+\.?\d*)').astype(float)

# Membuang baris yang gagal diubah jadi angka
df_clean = df_clean.dropna(subset=['Athlete average speed', 'Distance_num'])

# Memfilter anomali (Outlier) kecepatan dan jarak yang tidak logis
df_clean = df_clean[(df_clean['Athlete average speed'] >= 2.0) & (df_clean['Athlete average speed'] <= 25.0)] 
df_clean = df_clean[(df_clean['Distance_num'] > 0)]

# Membuat Label Target (Level)
def determine_level(row):
    speed = row['Athlete average speed']
    distance = row['Distance_num']
    if speed < 6.0 and distance <= 50: return 'Beginner'
    elif speed >= 8.0 or distance >= 100: return 'Advanced'
    else: return 'Intermediate'

df_clean['Level'] = df_clean.apply(determine_level, axis=1)

# Mengisolasi hanya kolom yang akan dipakai untuk pelatihan AI
fitur_final = ['Athlete age', 'Athlete average speed', 'Distance_num', 'Athlete gender', 'Level']
df_clean = df_clean[fitur_final]

# EDA(Exploratory Data Analysis)
print("\nExploratory data analysis...")

print("--- Ringkasan Statistik ---")
print(df_clean[['Athlete age', 'Athlete average speed', 'Distance_num']].describe())

print("\n--- Distribusi Level Pelari ---")
print(df_clean['Level'].value_counts())

os.makedirs('reports', exist_ok=True) 

plt.figure(figsize=(8, 5))
sns.countplot(data=df_clean, x='Level', palette='viridis', order=['Beginner', 'Intermediate', 'Advanced'])
plt.title('Distribusi Level Pelari dalam Dataset')
plt.ylabel('Jumlah Pelari')
plt.xlabel('Tingkat Kemampuan')
plt.tight_layout()
plt.savefig('reports/eda_distribusi_level.png')
print("Grafik EDA berhasil disimpan di folder 'reports/eda_distribusi_level.png'")


# Data Splitting(Pembagian Data)
print("\nData Splitting...")
train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean['Level'])
print(f"Data dipecah: {train_df.shape[0]} baris untuk Training, {test_df.shape[0]} baris untuk Testing.")

# Penyimpanan Data
print("\nMenyimpan hasil ke folder processed...")
os.makedirs('data/processed', exist_ok=True)

train_df.to_csv('data/processed/train.csv', index=False)
test_df.to_csv('data/processed/test.csv', index=False)

print("Data Preparation Selesai..........")