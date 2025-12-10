# Jalankan lokal:  streamlit run dashboard.py
# Untuk Streamlit Cloud: pastikan file data_kompetensi.csv ada di repo

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
st.write("Dashboard ML sederhana berbasis data kompetensi pegawai (tanpa upload, membaca CSV langsung).")


# -----------------------
# 3. Baca CSV dengan berbagai encoding (anti error Unicode)
# -----------------------

DATA_PATH = "data_kompetensi.csv"

df = None
for enc in ["utf-8", "latin1", "cp1252"]:
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc)
        break
    except Exception:
        continue

if df is None:
    st.error(
        f"Gagal membaca '{DATA_PATH}' menggunakan encoding utf-8 / latin1 / cp1252.\n"
        "Simpan ulang CSV sebagai UTF-8, lalu coba lagi."
    )
    st.stop()


# -----------------------
# 4. Tampilkan Data Awal
# -----------------------

st.subheader("👀 Sekilas Data")
st.dataframe(df.head())

st.markdown("---")


# -----------------------
# 5. Info Statistik Umum
# -----------------------

st.subheader("📌 Ringkasan Umum")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Pegawai", len(df))

with col2:
    st.metric("Jumlah Jabatan Unik", df["Jabatan"].nunique() if "Jabatan" in df else "-")

with col3:
    pel_col = "6. Jenis pelatihan"
    st.metric("Jenis Pelatihan Unik", df[pel_col].nunique() if pel_col in df else "-")


# -----------------------
# 6. Analisis Kompetensi (Top Words)
# -----------------------

st.subheader("🧠 Analisis Kompetensi yang Perlu Ditingkatkan")

kompetensi_col = "4. Kompetensi yang perlu ditingkatkan"

if kompetensi_col not in df.columns:
    st.error(f"Kolom '{kompetensi_col}' tidak ditemukan di CSV.")
else:
    top_words = get_top_words(df[kompetensi_col], n=20)
    if top_words:
        top_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])
        st.bar_chart(top_df.set_index("Kata"))
        st.table(top_df)
    else:
        st.write("Tidak ditemukan kata berarti pada kolom kompetensi.")


# -----------------------
# 7. Clustering Kompetensi (K-Means)
# -----------------------

st.subheader("🌀 Clustering Kompetensi (ML: K-Means)")

if kompetensi_col in df.columns:
    texts = df[kompetensi_col].fillna("").astype(str).apply(clean_text)

    if texts.str.len().sum() > 0:
        vectorizer = TfidfVectorizer(max_features=500)
        X = vectorizer.fit_transform(texts)

        k = st.slider("Jumlah Cluster", min_value=2, max_value=8, value=3)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

        df["Cluster_Kompetensi"] = kmeans.fit_predict(X)

        st.write("Distribusi cluster:")
        st.bar_chart(df["Cluster_Kompetensi"].value_counts())

        st.write("Contoh isi tiap cluster:")
        for cl in range(k):
            st.markdown(f"### Cluster {cl}")
            sample = df[df["Cluster_Kompetensi"] == cl].head(5)
            st.table(sample[["Nama", "Jabatan", kompetensi_col]])
    else:
        st.write("Data kompetensi terlalu sedikit untuk clustering.")


# -----------------------
# 8. Model Rekomendasi Pelatihan (Naive Bayes)
# -----------------------

st.subheader("🎯 Rekomendasi Jenis Pelatihan (ML: Naive Bayes)")

pelatihan_col = "6. Jenis pelatihan"

if kompetensi_col in df.columns and pelatihan_col in df.columns:

    data = df[[kompetensi_col, pelatihan_col]].dropna()

    if data[kompetensi_col].nunique() >= 3 and data[pelatihan_col].nunique() >= 2:

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
        st.write(f"Akurasi model: **{acc:.2f}**")

        user_input = st.text_area(
            "Masukkan kebutuhan kompetensi:",
            value="analisis data dan penulisan policy brief"
        )

        if st.button("🔮 Rekomendasikan Pelatihan"):
            X_user = vec_model.transform([clean_text(user_input)])
            pred = model.predict(X_user)[0]
            st.success(f"Rekomendasi pelatihan: **{pred}**")

    else:
        st.info("Data belum cukup untuk melatih model pelatihan.")

else:
    st.info("Kolom pelatihan atau kompetensi tidak ditemukan.")


# -----------------------
# 9. Ringkasan Otomatis Pegawai
# -----------------------

st.subheader("📝 Ringkasan Otomatis per Pegawai")

if "Nama" in df.columns and kompetensi_col in df.columns:

    pegawai = st.selectbox("Pilih Pegawai", df["Nama"].dropna().unique())

    row = df[df["Nama"] == pegawai].iloc[0]

    jab = row["Jabatan"] if "Jabatan" in row else "-"
    komp = row[kompetensi_col]
    hamb = row["5. Hambatan kompetensi"] if "5. Hambatan kompetensi" in row else "-"
    pel = row[pelatihan_col] if pelatihan_col in row else "-"

    st.markdown(f"""
### **Ringkasan Pegawai**

**Nama:** {pegawai}  
**Jabatan:** {jab}  

Kompetensi yang perlu ditingkatkan: **{komp}**  
Hambatan: **{hamb}**  
Pelatihan relevan: **{pel}**  

Disarankan fokus pada penguatan kompetensi inti dan mengikuti pelatihan terkait.
""")
else:
    st.info("Kolom 'Nama' atau kompetensi tidak ditemukan.")
