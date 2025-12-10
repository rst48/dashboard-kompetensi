# Jalankan lokal: streamlit run dashboard.py
# Di Streamlit Cloud: pastikan file data_kompetensi.csv (pakai ; sebagai pemisah) ada di repo yang sama.

import streamlit as st
import pandas as pd
import re
from collections import Counter

# ===============================
# 1. Fungsi bantu
# ===============================

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9àáâãäåèéêëìíîïòóôõöùúûüçñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

STOPWORDS_ID = set([
    "dan", "yang", "untuk", "dalam", "pada", "dengan", "agar", "dapat",
    "adalah", "atau", "di", "ke", "sebagai", "dari", "itu", "terhadap",
    "akan", "dalam", "yang", "saya", "dalam"
])

def get_top_words(series, n=15):
    all_words = []
    for text in series.fillna(""):
        text = clean_text(text)
        words = [w for w in text.split() if w not in STOPWORDS_ID and len(w) > 2]
        all_words.extend(words)
    counter = Counter(all_words)
    return counter.most_common(n)

# ===============================
# 2. Layout utama
# ===============================

st.set_page_config(page_title="Ringkasan CPNS PSEKP 2025", layout="wide")

st.title("📊 Dashboard Ringkasan Kompetensi CPNS PSEKP 2025")
st.write("Ringkasan kebutuhan kompetensi, hambatan, dan pelatihan dari data CPNS PSEKP 2025.")

# ===============================
# 3. Baca data CSV (pakai ; dan beberapa encoding)
# ===============================

DATA_PATH = "data_kompetensi.csv"

df = None
for enc in ["utf-8", "latin1", "cp1252"]:
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc, sep=";", quotechar='"')
        break
    except Exception:
        continue

if df is None:
    st.error(
        f"Gagal membaca '{DATA_PATH}' dengan encoding utf-8/latin1/cp1252.\n"
        "Coba simpan ulang CSV sebagai UTF-8 dan pastikan pemisahnya ';'."
    )
    st.stop()

# Rapikan nama kolom (hapus spasi di awal/akhir)
df.columns = df.columns.str.strip()

# ===============================
# 4. Ringkasan umum
# ===============================

st.subheader("📌 Ringkasan Umum CPNS PSEKP 2025")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Jumlah Peserta", len(df))

with col2:
    if "Jabatan" in df.columns:
        st.metric("Jabatan Unik", df["Jabatan"].nunique())
    else:
        st.metric("Jabatan Unik", "-")

with col3:
    if "KODE" in df.columns:
        st.metric("Kode Jabatan / JF Unik", df["KODE"].nunique())
    else:
        st.metric("Kode Jabatan / JF Unik", "-")

with col4:
    if "10. Metode pembelajaran" in df.columns:
        st.metric("Metode Belajar Unik", df["10. Metode pembelajaran"].nunique())
    else:
        st.metric("Metode Belajar Unik", "-")

st.markdown("---")

st.subheader("👀 Sekilas Data (Top 5)")
st.dataframe(df.head())

st.markdown("---")

# ===============================
# 5. Kompetensi yang paling banyak dibutuhkan
# ===============================

st.subheader("🧠 Kompetensi yang Paling Banyak Dibutuhkan")

kompetensi_col = "4. Kompetensi yang perlu ditingkatkan"

if kompetensi_col not in df.columns:
    st.error(f"Kolom '{kompetensi_col}' tidak ditemukan di CSV. Cek header file.")
else:
    # 5.1. Ringkasan per frasa (isi apa adanya)
    st.markdown("### 🔹 Top 10 Frasa Jawaban Kompetensi")
    value_counts = df[kompetensi_col].fillna("Tidak diisi").value_counts().head(10)
    st.bar_chart(value_counts)
    st.table(
        value_counts.reset_index().rename(
            columns={"index": "Frasa Kompetensi", kompetensi_col: "Jumlah Responden"}
        )
    )

    # 5.2. Ringkasan per kata (kata kunci terbesar)
    st.markdown("### 🔹 Top 20 Kata Kunci Kompetensi")
    top_words = get_top_words(df[kompetensi_col], n=20)
    if top_words:
        top_df = pd.DataFrame(top_words, columns=["Kata", "Frekuensi"])
        st.bar_chart(top_df.set_index("Kata"))
        st.table(top_df)
    else:
        st.write("Tidak ada teks kompetensi yang bisa dianalisis.")

# ===============================
# 6. Hambatan kompetensi
# ===============================

st.markdown("---")
st.subheader("🚧 Hambatan Kompetensi yang Sering Muncul")

hambatan_col = "5. Hambatan kompetensi"

if hambatan_col in df.columns:
    hambatan_counts = df[hambatan_col].fillna("Tidak diisi").value_counts().head(10)
    st.bar_chart(hambatan_counts)
    st.table(
        hambatan_counts.reset_index().rename(
            columns={"index": "Hambatan", hambatan_col: "Jumlah Responden"}
        )
    )
else:
    st.info("Kolom '5. Hambatan kompetensi' tidak ditemukan.")

# ===============================
# 7. Jenis pelatihan yang paling banyak dipilih
# ===============================

st.markdown("---")
st.subheader("🎓 Jenis Pelatihan yang Paling Banyak Dipilih")

pelatihan_col = "6. Jenis pelatihan"

if pelatihan_col in df.columns:
    pel_counts = df[pelatihan_col].fillna("Tidak diisi").value_counts().head(10)
    st.bar_chart(pel_counts)
    st.table(
        pel_counts.reset_index().rename(
            columns={"index": "Jenis Pelatihan", pelatihan_col: "Jumlah Responden"}
        )
    )
else:
    st.info("Kolom '6. Jenis pelatihan' tidak ditemukan.")

# ===============================
# 8. Metode pembelajaran yang diminati
# ===============================

st.markdown("---")
st.subheader("📚 Metode Pembelajaran yang Paling Diminati")

metode_col = "10. Metode pembelajaran"

if metode_col in df.columns:
    met_counts = df[metode_col].fillna("Tidak diisi").value_counts().head(10)
    st.bar_chart(met_counts)
    st.table(
        met_counts.reset_index().rename(
            columns={"index": "Metode Pembelajaran", metode_col: "Jumlah Responden"}
        )
    )
else:
    st.info("Kolom '10. Metode pembelajaran' tidak ditemukan.")

# ===============================
# 9. Ringkasan naratif otomatis
# ===============================

st.markdown("---")
st.subheader("📝 Ringkasan Naratif Otomatis CPNS PSEKP 2025")

summary_parts = []

# Kompetensi utama (kata kunci)
if kompetensi_col in df.columns:
    top_words = get_top_words(df[kompetensi_col], n=7)
    if top_words:
        top_comp = ", ".join([w for w, _ in top_words])
        summary_parts.append(f"- **Kompetensi yang paling sering muncul:** {top_comp}")

# Hambatan utama
if hambatan_col in df.columns:
    hambatan_top = df[hambatan_col].dropna().value_counts().head(3).index.tolist()
    if hambatan_top:
        summary_parts.append(f"- **Hambatan utama:** { '; '.join(hambatan_top) }")

# Pelatihan utama
if pelatihan_col in df.columns:
    pel_top = df[pelatihan_col].dropna().value_counts().head(3).index.tolist()
    if pel_top:
        summary_parts.append(f"- **Jenis pelatihan yang paling banyak dipilih:** { '; '.join(pel_top) }")

# Metode belajar utama
if metode_col in df.columns:
    met_top = df[metode_col].dropna().value_counts().head(2).index.tolist()
    if met_top:
        summary_parts.append(f"- **Metode belajar favorit:** { '; '.join(met_top) }")

if summary_parts:
    st.markdown("\n".join(summary_parts))
else:
    st.write("Belum cukup data untuk membuat ringkasan naratif.")

# ===============================
# 10. Detail per orang (opsional)
# ===============================

st.markdown("---")
st.subheader("🔎 Detail Jawaban per CPNS")

if "Nama" in df.columns:
    nama = st.selectbox("Pilih Nama", df["Nama"].dropna().unique())
    row = df[df["Nama"] == nama].iloc[0]

    st.markdown(f"### {nama}")
    jab = row.get("Jabatan", "-")
    st.markdown(f"**Jabatan:** {jab}")

    komp = row.get(kompetensi_col, "-")
    hamb = row.get(hambatan_col, "-")
    pel = row.get(pelatihan_col, "-")
    metode = row.get(metode_col, "-")

    st.markdown(f"- **Kompetensi yang perlu ditingkatkan:** {komp}")
    st.markdown(f"- **Hambatan kompetensi:** {hamb}")
    st.markdown(f"- **Jenis pelatihan yang diinginkan:** {pel}")
    st.markdown(f"- **Metode pembelajaran pilihan:** {metode}")
else:
    st.info("Kolom 'Nama' tidak ditemukan.")
