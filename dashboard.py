# Jalankan lokal: streamlit run dashboard.py
# Di Streamlit Cloud: pastikan file data_kompetensi.csv ada di repo yang sama.
# Format file mengikuti contoh yang kamu kirim: seluruh baris diapit tanda kutip, dipisah dengan ;

import streamlit as st
import pandas as pd
import re
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    "akan", "yang", "saya"
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
st.write("Ringkasan kebutuhan kompetensi, hambatan, pelatihan, dan tanya jawab berbasis data CPNS PSEKP 2025.")

# ===============================
# 3. Baca data CSV (format spesial)
# ===============================

DATA_PATH = "data_kompetensi.csv"

df = None
read_ok = False

# Pertama: baca apa adanya (pandas anggap default delimiter koma)
for enc in ["utf-8", "latin1", "cp1252"]:
    try:
        df = pd.read_csv(DATA_PATH, encoding=enc)
        read_ok = True
        break
    except Exception:
        continue

if not read_ok or df is None:
    st.error(
        f"Gagal membaca '{DATA_PATH}' dengan encoding utf-8/latin1/cp1252.\n"
        "Coba simpan ulang CSV sebagai UTF-8."
    )
    st.stop()

# Jika hasil baca cuma 1 kolom dan di dalam namanya masih ada ';',
# berarti seluruh baris masih jadi satu string → kita pecah manual.
if len(df.columns) == 1 and ";" in df.columns[0]:
    raw_header = df.columns[0]
    header_str = raw_header.strip().strip('"')
    header_cols = [h.strip() for h in header_str.split(";")]

    raw_rows = df.iloc[:, 0].astype(str)
    df = raw_rows.str.split(";", expand=True)
    df.columns = header_cols

# rapikan nama kolom
df.columns = df.columns.str.strip()

# nama kolom penting
KOMPETENSI_COL = "4. Kompetensi yang perlu ditingkatkan"
HAMBATAN_COL   = "5. Hambatan kompetensi"
PELATIHAN_COL  = "6. Jenis pelatihan"
METODE_COL     = "10. Metode pembelajaran"

# ===============================
# 4. Ringkasan umum
# ===============================

st.subheader("📌 Ringkasan Umum CPNS PSEKP 2025")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Jumlah Peserta", len(df))

with col2:
    st.metric("Jabatan Unik", df["Jabatan"].nunique() if "Jabatan" in df.columns else "-")

with col3:
    st.metric("Kode Jabatan / JF Unik", df["KODE"].nunique() if "KODE" in df.columns else "-")

with col4:
    st.metric("Metode Belajar Unik", df[METODE_COL].nunique() if METODE_COL in df.columns else "-")

st.markdown("---")

st.subheader("👀 Sekilas Data (Top 5)")
st.dataframe(df.head())

st.markdown("---")

# ===============================
# 5. Kompetensi yang paling banyak dibutuhkan
# ===============================

st.subheader("🧠 Kompetensi yang Paling Banyak Dibutuhkan")

if KOMPETENSI_COL not in df.columns:
    st.error(f"Kolom '{KOMPETENSI_COL}' tidak ditemukan di CSV. Nama kolom yang ada: {list(df.columns)}")
else:
    st.markdown("### 🔹 Top 10 Frasa Jawaban Kompetensi")
    value_counts = df[KOMPETENSI_COL].fillna("Tidak diisi").value_counts().head(10)
    st.bar_chart(value_counts)
    st.table(
        value_counts.reset_index().rename(
            columns={"index": "Frasa Kompetensi", KOMPETENSI_COL: "Jumlah Responden"}
        )
    )

    st.markdown("### 🔹 Top 20 Kata Kunci Kompetensi")
    top_words = get_top_words(df[KOMPETENSI_COL], n=20)
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

if HAMBATAN_COL in df.columns:
    hambatan_counts = df[HAMBATAN_COL].fillna("Tidak diisi").value_counts().head(10)
    st.bar_chart(hambatan_counts)
    st.table(
        hambatan_counts.reset_index().rename(
            columns={"index": "Hambatan", HAMBATAN_COL: "Jumlah Responden"}
        )
    )
else:
    st.info(f"Kolom '{HAMBATAN_COL}' tidak ditemukan.")

# ===============================
# 7. Jenis pelatihan yang paling banyak dipilih
# ===============================

st.markdown("---")
st.subheader("🎓 Jenis Pelatihan yang Paling Banyak Dipilih")

if PELATIHAN_COL in df.columns:
    pel_counts = df[PELATIHAN_COL].fillna("Tidak diisi").value_counts().head(10)
    st.bar_chart(pel_counts)
    st.table(
        pel_counts.reset_index().rename(
            columns={"index": "Jenis Pelatihan", PELATIHAN_COL: "Jumlah Responden"}
        )
    )
else:
    st.info(f"Kolom '{PELATIHAN_COL}' tidak ditemukan.")

# ===============================
# 8. Metode pembelajaran yang diminati
# ===============================

st.markdown("---")
st.subheader("📚 Metode Pembelajaran yang Paling Diminati")

if METODE_COL in df.columns:
    met_counts = df[METODE_COL].fillna("Tidak diisi").value_counts().head(10)
    st.bar_chart(met_counts)
    st.table(
        met_counts.reset_index().rename(
            columns={"index": "Metode Pembelajaran", METODE_COL: "Jumlah Responden"}
        )
    )
else:
    st.info(f"Kolom '{METODE_COL}' tidak ditemukan.")

# ===============================
# 9. Ringkasan naratif otomatis
# ===============================

st.markdown("---")
st.subheader("📝 Ringkasan Naratif Otomatis CPNS PSEKP 2025")

summary_parts = []

if KOMPETENSI_COL in df.columns:
    top_words = get_top_words(df[KOMPETENSI_COL], n=7)
    if top_words:
        top_comp = ", ".join([w for w, _ in top_words])
        summary_parts.append(f"- **Kompetensi yang paling sering muncul:** {top_comp}")

if HAMBATAN_COL in df.columns:
    hambatan_top = df[HAMBATAN_COL].dropna().value_counts().head(3).index.tolist()
    if hambatan_top:
        summary_parts.append(f"- **Hambatan utama:** { '; '.join(hambatan_top) }")

if PELATIHAN_COL in df.columns:
    pel_top = df[PELATIHAN_COL].dropna().value_counts().head(3).index.tolist()
    if pel_top:
        summary_parts.append(f"- **Jenis pelatihan yang paling banyak dipilih:** { '; '.join(pel_top) }")

if METODE_COL in df.columns:
    met_top = df[METODE_COL].dropna().value_counts().head(2).index.tolist()
    if met_top:
        summary_parts.append(f"- **Metode belajar favorit:** { '; '.join(met_top) }")

if summary_parts:
    st.markdown("\n".join(summary_parts))
else:
    st.write("Belum cukup data untuk membuat ringkasan naratif.")

# ===============================
# 10. Detail per CPNS
# ===============================

st.markdown("---")
st.subheader("🔎 Detail Jawaban per CPNS")

if "Nama" in df.columns:
    nama = st.selectbox("Pilih Nama", df["Nama"].dropna().unique())
    row = df[df["Nama"] == nama].iloc[0]

    st.markdown(f"### {nama}")
    jab = row.get("Jabatan", "-")
    st.markdown(f"**Jabatan:** {jab}")

    komp   = row.get(KOMPETENSI_COL, "-")
    hamb   = row.get(HAMBATAN_COL, "-")
    pel    = row.get(PELATIHAN_COL, "-")
    metode = row.get(METODE_COL, "-")

    st.markdown(f"- **Kompetensi yang perlu ditingkatkan:** {komp}")
    st.markdown(f"- **Hambatan kompetensi:** {hamb}")
    st.markdown(f"- **Jenis pelatihan yang diinginkan:** {pel}")
    st.markdown(f"- **Metode pembelajaran pilihan:** {metode}")
else:
    st.info("Kolom 'Nama' tidak ditemukan.")

# ===============================
# 11. Q&A: “AI” berbasis CSV
# ===============================

st.markdown("---")
st.subheader("🤖 Tanya Jawab Berbasis Data CPNS")

st.write(
    "Ketik pertanyaanmu (misal: *'kompetensi apa yang banyak dibutuhkan analis kebijakan?'* "
    "atau *'pelatihan apa yang sering muncul untuk PKSTI?'*). "
    "Jawaban diambil dari isi CSV ini, bukan dari internet."
)

def build_qa_index(df: pd.DataFrame):
    # Gabungkan beberapa kolom penting jadi satu teks per baris
    cols_candidate = []
    for col in ["Nama", "Jabatan", "KODE",
                KOMPETENSI_COL, HAMBATAN_COL, PELATIHAN_COL, METODE_COL]:
        if col in df.columns:
            cols_candidate.append(col)

    if not cols_candidate:
        return None, None, None

    sub = df[cols_candidate].fillna("").astype(str)
    # Gabungkan isi semua kolom menjadi satu string per baris
    corpus = sub.apply(lambda row: " ".join(row.values), axis=1)
    corpus_clean = corpus.apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(corpus_clean)

    return vectorizer, X, corpus_clean

vectorizer, X_corpus, corpus = build_qa_index(df)

if vectorizer is None:
    st.info("Belum bisa membuat indeks Q&A karena kolom-kolom penting tidak lengkap.")
else:
    question = st.text_input("Tulis pertanyaanmu di sini:")
    top_k = st.slider("Jumlah jawaban yang ditampilkan", 1, 5, 3)

    if st.button("🔍 Cari Jawaban"):
        if not question.strip():
            st.warning("Isi dulu pertanyaannya ya 😊")
        else:
            q_clean = clean_text(question)
            q_vec = vectorizer.transform([q_clean])
            sims = cosine_similarity(q_vec, X_corpus)[0]
            top_idx = sims.argsort()[::-1][:top_k]

            st.markdown("#### Hasil yang paling relevan:")

            for rank, idx in enumerate(top_idx, start=1):
                row = df.iloc[idx]
                skor = sims[idx]

                st.markdown(f"**#{rank} – Skor kemiripan: {skor:.2f}**")

                nama_r = row.get("Nama", "-")
                jab_r  = row.get("Jabatan", "-")
                kode_r = row.get("KODE", "-")
                komp_r = row.get(KOMPETENSI_COL, "-") if KOMPETENSI_COL in df.columns else "-"
                hamb_r = row.get(HAMBATAN_COL, "-")   if HAMBATAN_COL in df.columns else "-"
                pel_r  = row.get(PELATIHAN_COL, "-")  if PELATIHAN_COL in df.columns else "-"
                met_r  = row.get(METODE_COL, "-")     if METODE_COL in df.columns else "-"

                st.markdown(f"- **Nama:** {nama_r}")
                st.markdown(f"- **Jabatan / Kode:** {jab_r} ({kode_r})")
                st.markdown(f"- **Kompetensi yang perlu ditingkatkan:** {komp_r}")
                st.markdown(f"- **Hambatan kompetensi:** {hamb_r}")
                st.markdown(f"- **Jenis pelatihan yang diinginkan:** {pel_r}")
                st.markdown(f"- **Metode pembelajaran:** {met_r}")
                st.markdown("---")
