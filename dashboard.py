# Jalankan lokal:  streamlit run dashboard.py
# Di streamlit.io: pastikan file data_kompetensi.csv ada di repo yang sama

import streamlit as st
import pandas as pd
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# -----------------------
# 1. Fungsi bantu
# -----------------------

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9àáâãäåèéêëìíîïòóôõöùúûüçñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Stopwords sederhana (bisa kamu tambah sendiri)
STOPWORDS_ID = set([
    "dan", "yang", "untuk", "dalam", "pada", "dengan", "agar", "dapat",
    "adalah", "atau", "di", "ke", "sebagai", "dari", "itu", "terhadap",
    "yang", "akan", "dalam"
])

def get_top_words(series, n=15):
    all_words = []
    for text in series.fillna(""):
        text = clean_text(text)
        words = [w for w in text.split() if w not in STOPWORDS_ID and len(w) > 2]
        all_words.extend(words)
    counter = Counter(all_words)
    return counter.most_common(n)

# -----------------------
# 2. Layout Streamlit
# -----------------------

st.set_page_config(page_title="Dashboard Analisis Kompetensi", layout="wide")

st.title("📊 Dashboard Analisis Kompetensi & Pelatihan Pegawai")
st.write("Contoh dashboard ML sederhana berbasis data kompetensi pegawai (tanpa upload, langsung dari file CSV).")

# -----------------------
# 3. Baca data langsung dari file
# -----------------------

DATA_PATH = "data_kompetensi.csv"   # pastikan nama ini sama dengan file di repo

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"File '{DATA_PATH}' tidak ditemukan. Pastikan file CSV ada di repo dan namanya benar.")
    st.stop()

st.subheader("👀 Sekilas Data")
st.dataframe(df.head())

st.markdown("---")

# -----------------------
# 4. Info Umum & Statistik
# -----------------------

st.subheader("📌 Ringkasan Umum")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Pegawai", len(df))

with col2:
    if "Jabatan" in df.columns:
        st.metric("Jumlah Jabatan Unik", df["Jabatan"].nunique())
    else:
        st.metric("Jumlah Jabatan Unik", "-")

with col3:
    if "6. Jenis pelatihan" in df.columns:
        st.metric("Jenis Pelatihan Unik", df["6. Jenis pelatihan"].nunique())
    else:
        st.metric("Jenis Pelatihan Unik", "-")

# -----------------------
# 5. Analisis Kompetensi (Top Words)
# -----------------------

st.subheader("🧠 Analisis Kompetensi yang Perlu Ditingkatkan")

kompetensi_col = "4. Kompetensi yang perlu ditingkatkan"
if kompetensi_col not in df.columns:
    st.error(f"Kolom '{kompetensi_col}' tidak ditemukan di CSV. Sesuaikan nama kolom di kode.")
else:
    top_words = get_top_words(df[kompetensi_col], n=20)
    if not top_words:
        st.write("Tidak ada teks kompetensi yang bisa dianalisis.")
    else:
        # Ubah ke dataframe untuk ditampilkan sebagai bar chart
        top_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])
        st.bar_chart(top_df.set_index("Kata"))

        st.write("**Top kata yang sering muncul:**")
        st.table(top_df)

# -----------------------
# 6. Clustering Kompetensi (K-Means)
# -----------------------

st.subheader("🌀 Clustering Kompetensi (ML: K-Means)")

if kompetensi_col in df.columns:
    texts = df[kompetensi_col].fillna("").astype(str).apply(clean_text)
    if texts.str.len().sum() == 0:
        st.write("Data kompetensi kosong, tidak bisa dilakukan clustering.")
    else:
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(texts)

        k = st.slider("Pilih jumlah cluster", min_value=2, max_value=8, value=3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        df["Cluster_Kompetensi"] = kmeans.fit_predict(X)

        st.write("Distribusi pegawai per cluster:")
        st.bar_chart(df["Cluster_Kompetensi"].value_counts().sort_index())

        st.write("Contoh isi per cluster:")
        for cl in range(k):
            st.markdown(f"**Cluster {cl}**")
            sample = df[df["Cluster_Kompetensi"] == cl].head(5)
            if not sample.empty:
                st.table(sample[["Nama", "Jabatan", kompetensi_col]])
            else:
                st.write("_(Cluster kosong)_")

# -----------------------
# 7. Model Sederhana Rekomendasi Jenis Pelatihan
# -----------------------

st.subheader("🎯 Rekomendasi Jenis Pelatihan (ML: Naive Bayes)")

pelatihan_col = "6. Jenis pelatihan"

if kompetensi_col in df.columns and pelatihan_col in df.columns:
    data = df[[kompetensi_col, pelatihan_col]].dropna()
    if data[kompetensi_col].nunique() < 3 or data[pelatihan_col].nunique() < 2:
        st.write("Data belum cukup bervariasi untuk melatih model (butuh lebih banyak baris & variasi).")
    else:
        X_text = data[kompetensi_col].astype(str).apply(clean_text)
        y = data[pelatihan_col].astype(str)

        vec_model = TfidfVectorizer(max_features=1000)
        X_vec = vec_model.fit_transform(X_text)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vec, y, test_size=0.2, random_state=42
        )

        model = MultinomialNB()
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        st.write(f"Akurasi model sederhana di data test: **{acc:.2f}** (ini cuma baseline ya 😄)")

        st.markdown("### Coba masukkan kebutuhan kompetensi, lalu model akan merekomendasikan jenis pelatihan:")

        user_input = st.text_area(
            "Tuliskan kompetensi yang perlu ditingkatkan / tugas yang dihadapi:",
            value="analisis data dan penulisan policy brief"
        )

        if st.button("🔮 Rekomendasikan Pelatihan"):
            if user_input.strip():
                X_user = vec_model.transform([clean_text(user_input)])
                pred = model.predict(X_user)[0]
                st.success(f"Rekomendasi jenis pelatihan: **{pred}**")
            else:
                st.warning("Tolong isi dulu teks kompetensinya 😊")
else:
    st.info(f"Pastikan kolom '{kompetensi_col}' dan '{pelatihan_col}' ada di file CSV.")

# -----------------------
# 8. Ringkasan Otomatis per Pegawai (Template)
# -----------------------

st.subheader("📝 Ringkasan Otomatis per Pegawai (Template)")

if kompetensi_col in df.columns:
    if "Nama" in df.columns:
        nama_list = df["Nama"].dropna().unique().tolist()
        selected_nama = st.selectbox("Pilih pegawai", options=nama_list)

        row = df[df["Nama"] == selected_nama].iloc[0]

        jabatan = row["Jabatan"] if "Jabatan" in row else "-"
        komp = row[kompetensi_col] if kompetensi_col in row else "-"
        hambatan = row["5. Hambatan kompetensi"] if "5. Hambatan kompetensi" in row else "-"
        pelatihan = row[pelatihan_col] if pelatihan_col in row else "-"

        st.markdown("#### Ringkasan:")
        summary_text = f"""
**Nama:** {selected_nama}  
**Jabatan:** {jabatan}  

Pegawai ini membutuhkan peningkatan kompetensi pada: **{komp}**.  
Hambatan utama yang dihadapi adalah: **{hambatan}**.  
Jenis pelatihan yang dipilih/direkomendasikan: **{pelatihan}**.  

Disarankan untuk fokus pada penguatan kompetensi inti tersebut melalui pelatihan yang relevan 
dan pendampingan yang sesuai dengan tugas jabatan sehari-hari.
"""
        st.markdown(summary_text)
    else:
        st.info("Kolom 'Nama' tidak ditemukan, sesuaikan di kode jika nama kolomnya beda.")
