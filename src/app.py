import streamlit as st
import pandas as pd
import joblib
import os

# Logika Rule-Based
class SportScienceRecommender:
    """Logika berdasarkan Hal Higdon, Jack Daniels, dan Pedoman Medis Olahraga Usia"""
    
    def _hitung_pace(self, menit, detik, modifier_detik):
        total_detik = (menit * 60) + detik + modifier_detik
        total_detik = max(total_detik, 210) 
        hasil_menit = total_detik // 60
        hasil_detik = total_detik % 60
        return f"{hasil_menit:02d}:{hasil_detik:02d} /km"

    def dapatkan_rekomendasi(self, umur, level_ai, target_lari, pace_m, pace_d, mileage, hari):
        level_ai = level_ai.capitalize()
        
        # Peringatan umur untuk anak-anak
        if umur < 18 and target_lari in ['Full Marathon (42K)', 'Ultra Running (50K+)']:
            return {
                "status": "warning",
                "program": "PEMBATASAN MEDIS (PEDIATRIK)",
                "referensi": "Pedoman Kedokteran Olahraga Remaja (Youth Sports Guidelines)",
                "catatan": f"Usia Anda ({umur} tahun) masih dalam masa pertumbuhan tulang. Lari Marathon/Ultra berisiko merusak lempeng pertumbuhan tulang (epiphyseal plates) secara permanen.",
                "jadwal": {
                    "Instruksi Medis": "Sistem menolak memberikan jadwal 42K/50K. Fokuslah berlatih kecepatan (Speedwork) di jarak 5K hingga 10K sampai usia Anda mencapai minimal 18 tahun."
                }
            }

        # Peringatan umur untuk lansia
        catatan_lansia = ""
        if umur >= 60:
            if target_lari in ['Full Marathon (42K)', 'Ultra Running (50K+)'] or level_ai == 'Advanced':
                return {
                    "status": "warning",
                    "program": "PERINGATAN KARDIOVASKULAR (SENIOR)",
                    "referensi": "American Heart Association - Senior Athlete",
                    "catatan": f"Di usia {umur} tahun, latihan intensitas tinggi (Advanced) atau jarak ekstrem sangat membebani kerja jantung dan persendian.",
                    "jadwal": {
                        "Instruksi 1": "WAJIB mendapatkan Medical Clearance (Izin Dokter Kardiologi) sebelum melakukan Marathon/Ultra.",
                        "Instruksi 2": "Sistem menyarankan Anda untuk menurunkan target ke Half-Marathon (21K) atau 10K demi menjaga kebugaran tanpa risiko cedera fatal."
                    }
                }
            else:
                # Jika lansia lari 5K/10K (Aman, tapi beri catatan tambahan)
                catatan_lansia = "\n\n **CATATAN SENIOR:** Tambahkan waktu pemanasan 10 menit ekstra. Pemulihan (recovery) otot Anda membutuhkan waktu lebih lama, jangan paksakan lari jika sendi terasa nyeri."

        # Validasi jarak mingguan(mileage)
        if target_lari == 'Half-Marathon (21K)' and mileage < 15:
            return self._peringatan_medis(target_lari, mileage, 20, "Easy Run 5km")
        elif target_lari == 'Full Marathon (42K)' and mileage < 30:
            return self._peringatan_medis(target_lari, mileage, 40, "Long Run 15km")
        elif target_lari == 'Ultra Running (50K+)' and mileage < 50:
            return self._peringatan_medis(target_lari, mileage, 60, "Back-to-Back Long Run")

        # Peringatan ekstrem untuk beginner dilarang ikut ultra
        if level_ai == 'Beginner' and target_lari == 'Ultra Running (50K+)':
            return {
                "status": "warning",
                "program": "DITOLAK SECARA MEDIS",
                "referensi": "ITRA (International Trail Running Association)",
                "catatan": "Sebagai pelari Beginner, mendaftar Ultra berisiko Rhabdomyolysis (kerusakan otot fatal hingga gagal ginjal).",
                "jadwal": {
                    "Instruksi": "Selesaikan program 21K dan 42K secara bertahap selama 1 tahun ke depan sebelum menyentuh Ultra."
                }
            }

        # Kalkulasi Target Pace Latihan
        pace_easy = self._hitung_pace(pace_m, pace_d, 60)
        pace_tempo = self._hitung_pace(pace_m, pace_d, -20)
        pace_long = self._hitung_pace(pace_m, pace_d, 45)
        pace_mp = self._hitung_pace(pace_m, pace_d, 15)

        # Rules untuk beginner
        if level_ai == 'Beginner':
            if target_lari in ['5K', '10K', 'Half-Marathon (21K)']:
                dist_long = "5km" if target_lari == '5K' else ("8km" if target_lari == '10K' else "14km")
                return {
                    "status": "success",
                    "program": f"Novice {target_lari} Preparation",
                    "referensi": "Hal Higdon's Novice Program",
                    "catatan": "Fokus pada konsistensi. Gunakan metode jalan-lari (Walk-Run) jika perlu." + catatan_lansia,
                    "jadwal": {
                        "Latihan 1": f"Easy Run 3km (Pace: {pace_easy})",
                        "Latihan 2": f"Easy Run 4km (Pace: {pace_easy})",
                        "Latihan 3": f"Long Run {dist_long} (Pace sangat lambat)",
                        "Istirahat": "Wajib istirahat atau cross-training ringan."
                    }
                }
            elif target_lari == 'Full Marathon (42K)':
                return {
                    "status": "success",
                    "program": "Marathon Survival (Novice 1)",
                    "referensi": "Hal Higdon's Novice 1 Marathon",
                    "catatan": "Tujuan utama MENCAPAI GARIS FINIS, bukan mengejar waktu." + catatan_lansia,
                    "jadwal": {
                        "Latihan 1": f"Easy Run 5km (Pace: {pace_easy})",
                        "Latihan 2": f"Easy Run 8km (Pace: {pace_easy})",
                        "Latihan 3": f"Easy Run 5km (Pace: {pace_easy})",
                        "Latihan 4": "Long Run 20-32km (Bawa gel energi dan air)"
                    }
                }

        # Rules untuk intermediate
        elif level_ai == 'Intermediate':
            if target_lari == 'Full Marathon (42K)':
                return {
                    "status": "success",
                    "program": "Intermediate Marathon (Pace Focus)",
                    "referensi": "Jack Daniels' Marathon Training",
                    "catatan": "Biasakan berlari di target Marathon Pace saat kaki sudah lelah." + catatan_lansia,
                    "jadwal": {
                        "Latihan 1": f"Easy Run 8km (Pace: {pace_easy})",
                        "Latihan 2": f"Tempo: 2km Pemanasan + 8km Pace {pace_mp} + 2km Pendinginan",
                        "Latihan 3": f"Easy Run 8km (Pace: {pace_easy})",
                        "Latihan 4": f"Long Run 25-34km (Sisipkan 5km Pace {pace_mp} di akhir)"
                    }
                }
            elif target_lari == 'Ultra Running (50K+)':
                return {
                    "status": "success",
                    "program": "Ultra Base & Time-on-Feet",
                    "referensi": "Relentless Forward Progress (Bryon Powell)",
                    "catatan": "Kunci Ultra: makan/minum sambil bergerak & adaptasi tanjakan." + catatan_lansia,
                    "jadwal": {
                        "Latihan 1": f"Easy Trail Run 10km (Cari rute menanjak)",
                        "Latihan 2": f"Tempo Run 8km di aspal (Pace: {pace_tempo})",
                        "Latihan 3": "Back-to-Back Long Run 1: 25km di hari Sabtu",
                        "Latihan 4": "Back-to-Back Long Run 2: 15km di hari Minggu (Kaki lelah)"
                    }
                }
            else:
                dist_long = "12km" if target_lari == '10K' else ("18km" if target_lari == 'Half-Marathon (21K)' else "8km")
                return {
                    "status": "success",
                    "program": f"Intermediate {target_lari} Speed & Endurance",
                    "referensi": "Jack Daniels' Phase II Training",
                    "catatan": "Mulai melibatkan Tempo Run untuk kekuatan ambang laktat." + catatan_lansia,
                    "jadwal": {
                        "Latihan 1": f"Easy Run 6km (Pace: {pace_easy})",
                        "Latihan 2": f"Tempo Run 5-8km (Pace: {pace_tempo})",
                        "Latihan 3": f"Easy Run 6km (Pace: {pace_easy})",
                        "Latihan 4": f"Long Run {dist_long} (Pace: {pace_long})"
                    }
                }

        # Rules untuk advanced
        elif level_ai == 'Advanced':
            if target_lari == 'Full Marathon (42K)':
                return {
                    "status": "success",
                    "program": "Advanced Marathon (BQ / Sub-3 Attempt)",
                    "referensi": "Pfitzinger Advanced Marathoning",
                    "catatan": "Volume sangat tinggi. Prioritaskan tidur 8 jam dan nutrisi pemulihan." + catatan_lansia,
                    "jadwal": {
                        "Latihan 1": f"Recovery Run 10km (Pace > {pace_easy})",
                        "Latihan 2": f"Lactate Threshold: 12km (Pace {pace_tempo})",
                        "Latihan 3": f"Medium-Long Run 18km (Pace {pace_long})",
                        "Latihan 4": f"Marathon Long Run: 35km (10km awal easy, 20km tengah Pace {pace_mp})"
                    }
                }
            elif target_lari == 'Ultra Running (50K+)':
                return {
                    "status": "success",
                    "program": "Competitive Ultra / 100K Preparation",
                    "referensi": "Jason Koop's Training for Ultrarunning",
                    "catatan": "Fokus ketahanan otot di turunan tajam (Eccentric loading) & VO2 Max." + catatan_lansia,
                    "jadwal": {
                        "Latihan 1": f"VO2 Max: 4 x 3 menit menanjak tajam (Rest jalan turun)",
                        "Latihan 2": f"Steady State Trail Run 15km",
                        "Latihan 3": "Back-to-Back 1: 35km Trail Run (+/- 1000m Elevasi)",
                        "Latihan 4": "Back-to-Back 2: 25km Trail Run di hari berikutnya"
                    }
                }
            else:
                return {
                    "status": "success",
                    "program": f"Advanced {target_lari} Peak Performance",
                    "referensi": "Jack Daniels' VDOT Formula",
                    "catatan": "Intensitas tinggi. Perhatikan hidrasi selama latihan." + catatan_lansia,
                    "jadwal": {
                        "Latihan 1": f"Interval: 6x800m (Pace sangat cepat) / Rest 2'",
                        "Latihan 2": f"Threshold Run 10km (Pace: {pace_tempo})",
                        "Latihan 3": f"Easy Run 12km (Pace: {pace_easy})",
                        "Latihan 4": f"Long Run 21-25km (Pace: {pace_long})"
                    }
                }

    def _peringatan_medis(self, target, mileage, target_mileage, saran_latihan):
        return {
            "status": "warning",
            "program": f"Pre-Conditioning {target} (Base Building)",
            "referensi": "Prinsip Adaptasi Jaringan Ikat & Kardio",
            "catatan": f"Jarak mingguan Anda ({mileage}km) terlalu rendah untuk {target}. Risiko cedera tulang/sendi sangat tinggi.",
            "jadwal": {
                "Fokus Utama": f"Tingkatkan jarak mingguan bertahap hingga minimal {target_mileage}km sebelum memulai program ini.",
                "Saran Latihan": f"Perbanyak frekuensi lari santai dan biasakan {saran_latihan} di akhir pekan."
            }
        }
# Input
st.set_page_config(page_title="AI Coach Lari", page_icon="🏃‍♂️", layout="wide")

st.title("REKOMENDASI PROGRAM LATIHAN LARI BERBASIS AI")
st.write("Sistem ini menggabungkan **Machine Learning** untuk menilai profil Anda dan **Sistem Pakar** untuk meresepkan jadwal.")

with st.sidebar:
    st.header("Profil Fisik (Input AI)")
    umur = st.number_input("Umur", 15, 80, 25)
    gender_input = st.selectbox("Gender", ["Pria", "Wanita"])
    gender = 'M' if gender_input == "Pria" else 'F'
    speed = st.number_input("Kecepatan Rata-rata (km/jam)", 2.0, 20.0, 6.0)
    distance = st.number_input("Jarak Sekali Lari (km)", 1.0, 100.0, 5.0)

# Target dan Latihan
st.subheader("Detail Latihan & Target")
col1, col2, col3 = st.columns(3)

with col1:
    target = st.selectbox("Target Lari Berikutnya",["5K", "10K", "Half-Marathon (21K)", "Full Marathon (42K)", "Ultra Running (50K+)"])
    mileage = st.number_input("Total Jarak Seminggu Ini (km)", 0, 200, 10)

with col2:
    st.write("Rata-rata Pace Sekarang (menit/km)")
    c_min, c_sec = st.columns(2)
    pace_m = c_min.number_input("Menit", 3, 15, 7)
    pace_d = c_sec.number_input("Detik", 0, 59, 30)

with col3:
    hari_aktif = st.slider("Hari Latihan per Minggu", 1, 6, 3)


# Output
if st.button("ANALISIS & BUAT JADWAL LATIHAN", use_container_width=True):
    
    # Load Model
    try:
        model = joblib.load('models/model_rekomendasi.pkl')
    except:
        st.error("Model AI tidak ditemukan! Silakan jalankan 'python src/model_training.py' terlebih dahulu.")
        st.stop()

    # Prediksi AI
    input_data = pd.DataFrame({
        'Athlete age': [umur],
        'Athlete average speed': [speed],
        'Distance_num': [distance],
        'Athlete gender': [gender]
    })
    predicted_level = model.predict(input_data)[0]

    # Ambil Rekomendasi Rule-Based
    coach = SportScienceRecommender()
    rekom = coach.dapatkan_rekomendasi(umur, predicted_level, target, pace_m, pace_d, mileage, hari_aktif)

    # Tampilkan Hasil
    st.divider()
    
    # Layout Hasil
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric("Hasil Analisis AI (Level)", predicted_level.upper())
        st.info(f"**Program:** {rekom['program']}\n\n**Referensi Teori:** {rekom['referensi']}")

    with res_col2:
        if rekom['status'] == 'warning':
            st.warning(f"**PERINGATAN KEAMANAN:** {rekom['catatan']}")
        else:
            st.success(f"**CATATAN PELATIH:** {rekom['catatan']}")
        
        st.markdown("#### Jadwal Latihan Mingguan Anda:")
        for hari, kegiatan in rekom['jadwal'].items():
            st.write(f"**{hari}:** {kegiatan}")
    
    st.balloons()