# REKOMENDASI PROGRAM LATIHAN LARI BERBASIS AI

Proyek Akhir Data Mining - Sistem cerdas untuk merekomendasikan jadwal latihan lari. Sistem ini menggabungkan prediksi **Machine Learning (Random Forest)** dengan logika medis olahraga **Rule-Based (Hal Higdon & Jack Daniels VDOT)**.

## Arsitektur MLOps (DAG)
Proyek ini menggunakan DVC untuk memisahkan alur *Data Engineering* dan *Model Training*. Berikut adalah alur pipeline-nya:

```mermaid
flowchart TD
    node1["prepare
    (src/data_preparation.py)"]
    node2["train
    (src/model_training.py)"]
    node1-->node2
```

## Cara Menjalankan Aplikasi Web
1. Pastikan model sudah dilatih dengan menjalankan `dvc repro`.
2. Jalankan perintah ini di terminal: `streamlit run src/app.py`