import startup

import json
import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE, "model")
HASIL_DIR = os.path.join(BASE, "hasil")
PANEL = os.path.join(BASE, "data", "processed", "dataset_panel.csv")

HEX = {"rendah": "#2ea043", "sedang": "#f0883e", "tinggi": "#e5534b"}

st.set_page_config(page_title="Kerawanan Karhutla", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
.block-container {padding-top: 3rem; max-width: 1100px;}
[data-testid="stAppDeployButton"], .stDeployButton {display: none;}
h1 {font-weight: 700; letter-spacing: -0.5px;}
.hasil-card {border-radius: 14px; padding: 1.1rem 1.4rem; border: 1px solid rgba(128,128,128,.28);}
.hasil-label {font-size: .8rem; opacity: .7; letter-spacing: .06em; text-transform: uppercase;}
.hasil-nilai {font-size: 2.8rem; font-weight: 800; line-height: 1.15;}
.hasil-ket {font-size: .85rem; opacity: .75; margin-top: .35rem;}
.rendah {background: rgba(46,160,67,.16); border-left: 6px solid #2ea043;}
.sedang {background: rgba(240,136,62,.18); border-left: 6px solid #f0883e;}
.tinggi {background: rgba(229,83,75,.18); border-left: 6px solid #e5534b;}
.kecil {font-size: .85rem; opacity: .75;}
.stApp {background: linear-gradient(180deg, rgba(20,83,45,.07) 0, transparent 340px);}
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {display: none;}
[data-testid="stRadio"] [role="radiogroup"] {gap: .5rem; flex-wrap: wrap;}
[data-testid="stRadio"] label {padding: .45rem 1.1rem; border-radius: 999px; cursor: pointer;
    background: rgba(15,118,110,.12); border: 1px solid rgba(15,118,110,.35);}
[data-testid="stRadio"] label > div:first-child {display: none;}
[data-testid="stRadio"] label:has(input:checked) {background: linear-gradient(120deg,#14532d,#0f766e); border-color: transparent;}
[data-testid="stRadio"] label:has(input:checked) * {color: #fff !important; font-weight: 700;}
.hero {border-radius: 18px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem; color: #fff;
       background: linear-gradient(120deg, #14532d 0%, #0f766e 55%, #2b9a6b 100%);
       box-shadow: 0 6px 20px rgba(15,118,110,.25);}
.hero .judul {font-size: 2.1rem; font-weight: 800; line-height: 1.2; letter-spacing: -.5px;}
.hero .sub {font-size: .95rem; opacity: .9; margin-top: .45rem; max-width: 760px;}
h3 {border-left: 5px solid #0f766e; padding-left: .7rem !important;}
</style>
""",
    unsafe_allow_html=True,
)

@st.cache_resource
def muat_model():
    return joblib.load(os.path.join(MODEL_DIR, "model_rf.joblib"))


@st.cache_data
def muat_meta():
    return json.load(open(os.path.join(MODEL_DIR, "meta.json"), encoding="utf-8"))


@st.cache_data
def muat_profil():
    return pd.read_csv(os.path.join(MODEL_DIR, "profil_provinsi.csv")).set_index("provinsi")


@st.cache_data
def muat_panel():
    df = pd.read_csv(PANEL)
    df.loc[~df["data_sah"], "luas_ha"] = np.nan  # Papua sebelum pemekaran: bukan nol, tapi tidak ada data
    return df



def hero(judul, sub=""):
    st.markdown(f'<div class="hero"><div class="judul">{judul}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def kelas_dari_luas(luas, ambang):
    if pd.isna(luas):
        return None
    return "rendah" if luas < ambang[0] else ("sedang" if luas <= ambang[1] else "tinggi")


def cek_berkas():
    perlu = [os.path.join(MODEL_DIR, "model_rf.joblib"), os.path.join(MODEL_DIR, "meta.json"),
             os.path.join(MODEL_DIR, "profil_provinsi.csv"), PANEL]
    hilang = [p for p in perlu if not os.path.exists(p)]
    if hilang:
        st.error("Berkas model belum ada. Jalankan dulu `python src/pipeline.py` dari folder KarhutlaAI.")
        st.code("\n".join(hilang))
        st.stop()


cek_berkas()
model, meta, profil, panel = muat_model(), muat_meta(), muat_profil(), muat_panel()
KELAS, AMBANG = meta["kelas"], meta["ambang_ha"]


def halaman_prediksi():
    hero("Prediksi Kelas Kerawanan Karhutla",
         "Perkiraan kelas kerawanan sebuah provinsi dalam satu tahun, berdasarkan riwayat luas karhutla dan curah hujan musim kemarau (Juli-Oktober) yang Anda masukkan.")

    prov = st.selectbox("Provinsi", list(profil.index), index=list(profil.index).index("Riau")
                        if "Riau" in profil.index else 0)
    p = profil.loc[prov]

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.subheader("Riwayat karhutla")
        luas1 = st.number_input("Luas karhutla tahun lalu (ha)", min_value=0.0,
                                value=float(round(p["luas_ha_terakhir"], 2)), step=100.0)
        luas2 = st.number_input("Luas karhutla dua tahun lalu (ha)", min_value=0.0,
                                value=float(round(np.expm1(p["log_prev"]), 2)), step=100.0)
    with c2, st.container(border=True):
        st.subheader("Curah hujan")
        ch_k = st.number_input("Total curah hujan musim kemarau Jul-Okt (mm)", min_value=0.0,
                               value=float(round(p["ch_kemarau_ref_mean"], 1)), step=10.0)
        ch_t = st.number_input("Total curah hujan setahun (mm)", min_value=0.0,
                               value=float(round(p["ch_tahunan_ref_mean"], 1)), step=50.0)

    std = p["ch_kemarau_ref_std"] if p["ch_kemarau_ref_std"] > 0 else np.nan
    z = (ch_k - p["ch_kemarau_ref_mean"]) / std
    if np.isnan(z):
        z = 0.0

    def susun(zv, chk):
        return pd.DataFrame([{
            "lag1_log_luas": np.log1p(luas1),
            "mean_lag_log_luas": np.mean([np.log1p(luas1), np.log1p(luas2)]),
            "ch_kemarau_mm": chk, "ch_kemarau_z": zv, "ch_tahunan_mm": ch_t,
            "gambut": p["gambut"], "has_gambut": p["has_gambut"],
            "iku_prov": p["iku_prov"], "iprlh": p["iprlh"],
        }])[meta["fitur"]]

    proba = model.predict_proba(susun(z, ch_k))[0]
    idx = int(np.argmax(proba))
    hasil = KELAS[idx]

    st.write("")
    a, b = st.columns([1, 1.4])
    with a:
        if z <= -1:
            ket = f"Kemarau {abs(z):.1f} simpangan baku LEBIH KERING dari normal provinsi ini."
        elif z >= 1:
            ket = f"Kemarau {z:.1f} simpangan baku LEBIH BASAH dari normal provinsi ini."
        else:
            ket = f"Curah hujan kemarau mendekati normal (z = {z:+.2f})."
        batas = (f"Ambang: rendah &lt; {AMBANG[0]:,} ha | sedang {AMBANG[0]:,}-{AMBANG[1]:,} ha | "
                 f"tinggi &gt; {AMBANG[1]:,} ha").replace(",", ".")
        st.markdown(
            f'<div class="hasil-card {hasil}"><div class="hasil-label">Kelas kerawanan - {prov}</div>'
            f'<div class="hasil-nilai">{hasil.upper()}</div>'
            f'<div class="hasil-ket">{ket}</div></div>'
            f'<div class="kecil" style="margin-top:.6rem">{batas}</div>',
            unsafe_allow_html=True,
        )
    with b:
        dfp = pd.DataFrame({"kelas": KELAS, "peluang": proba})
        bar = alt.Chart(dfp).mark_bar(cornerRadiusEnd=5, size=26).encode(
            x=alt.X("peluang:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title="Peluang")),
            y=alt.Y("kelas:N", sort=KELAS, title=None),
            color=alt.Color("kelas:N", scale=alt.Scale(domain=KELAS, range=[HEX[k] for k in KELAS]), legend=None),
            tooltip=["kelas", alt.Tooltip("peluang:Q", format=".1%")],
        )
        teks = alt.Chart(dfp).mark_text(align="left", dx=5, fontWeight="bold").encode(
            x="peluang:Q", y=alt.Y("kelas:N", sort=KELAS), text=alt.Text("peluang:Q", format=".0%"))
        st.altair_chart((bar + teks).properties(height=150), width="stretch")

    st.subheader("Skenario: bagaimana jika kemaraunya berbeda?")
    rows = []
    for nama, zv in [("Sangat kering (z = -1,5)", -1.5), ("Normal (z = 0)", 0.0), ("Sangat basah (z = +1,5)", 1.5)]:
        chk = max(0.0, p["ch_kemarau_ref_mean"] + zv * (0 if np.isnan(std) else std))
        pr = model.predict_proba(susun(zv, chk))[0]
        rows.append({"Skenario": nama, "Kemarau (mm)": round(chk, 0), "Kelas": KELAS[int(np.argmax(pr))],
                     **{f"P({k})": round(float(v), 2) for k, v in zip(KELAS, pr)}})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Jika kelas tidak berubah antar skenario, berarti model lebih bergantung pada riwayat karhutla "
               "daripada curah hujan untuk provinsi ini.")

    st.info(
        "Hasil ini bersifat indikatif dan untuk keperluan pembelajaran. Pada uji 2024-2025, performa model "
        "setara dengan tebakan sederhana \"kelas sama seperti tahun lalu\" (lihat halaman Performa Model)."
    )


def halaman_performa():
    hero("Performa Model", "Seberapa baik model menebak kelas kerawanan pada data uji 2024-2025, dibandingkan baseline sederhana.")
    tab = os.path.join(HASIL_DIR, "tabel_perbandingan_model.csv")
    if not os.path.exists(tab):
        st.warning("Hasil belum ada. Jalankan `python src/pipeline.py`.")
        return
    t = pd.read_csv(tab).drop(columns=["param"], errors="ignore")
    st.subheader("Perbandingan model (uji temporal 2024-2025)")
    st.dataframe(t.round(3), hide_index=True, width="stretch")
    st.caption("Model dilatih pada 2019-2023 dan diuji pada 2024-2025. "
               "`f1_cv_antar_provinsi` = validasi silang yang menguji ke provinsi yang tidak dilihat saat latih.")

    c1, c2 = st.columns(2)
    for col, berkas, judul in [(c1, "confusion_matrix.png", "Confusion matrix"),
                               (c2, "feature_importance.png", "Feature importance")]:
        pth = os.path.join(HASIL_DIR, berkas)
        with col:
            st.subheader(judul)
            if os.path.exists(pth):
                st.image(pth, width="stretch")

    ring = os.path.join(HASIL_DIR, "ringkasan.txt")
    if os.path.exists(ring):
        with st.expander("Ringkasan lengkap (termasuk uji perbandingan dengan baseline)"):
            st.code(open(ring, encoding="utf-8").read().split("CATATAN KETERBATASAN")[0].rstrip())


def halaman_data():
    hero("Data Historis Karhutla", "Luas karhutla per provinsi 2018-2025 (SiPongi) beserta curah hujan musim kemaraunya.")
    prov = st.selectbox("Provinsi", sorted(panel["provinsi"].unique()), key="prov_data")
    d = panel[panel["provinsi"] == prov].sort_values("tahun")
    d = d.assign(kelas_hist=d["luas_ha"].map(lambda v: kelas_dari_luas(v, AMBANG)))
    st.subheader(f"Luas karhutla {prov} (ha)")
    skala = alt.Scale(type="symlog")
    garis = alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=60, color="#2b7a78", filled=True), color="#2b7a78", strokeWidth=2.5).encode(
        x=alt.X("tahun:O", title="Tahun"), y=alt.Y("luas_ha:Q", scale=skala, title="Luas (ha, skala simlog)",
                axis=alt.Axis(values=[0, 100, 1000, 10000, 100000, 1000000], format=",.0f")),
        tooltip=["tahun", alt.Tooltip("luas_ha:Q", format=",.0f"), "kelas_hist"])
    ambang = alt.Chart(pd.DataFrame({"luas_ha": AMBANG})).mark_rule(strokeDash=[5, 5], color="#999").encode(
        y=alt.Y("luas_ha:Q", scale=skala))
    st.altair_chart((garis + ambang).properties(height=300), width="stretch")
    st.caption(f"Garis putus-putus = ambang kelas ({AMBANG[0]:,} ha dan {AMBANG[1]:,} ha).".replace(",", "."))
    st.dataframe(d[["tahun", "luas_ha", "kelas_hist", "ch_kemarau_mm", "ch_kemarau_z"]]
                 .rename(columns={"kelas_hist": "kelas", "ch_kemarau_mm": "kemarau_mm", "ch_kemarau_z": "kemarau_z"})
                 .round(2), hide_index=True, width="stretch")
    if prov in ("Papua", "Papua Barat Daya", "Papua Pegunungan", "Papua Selatan", "Papua Tengah"):
        st.caption("Data sebelum 2023 dikosongkan karena batas wilayah berubah akibat pemekaran Papua.")

    st.divider()
    thn = st.select_slider("Peringkat provinsi pada tahun", sorted(panel["tahun"].unique()), value=2025)
    r = panel[panel["tahun"] == thn].dropna(subset=["luas_ha"]).nlargest(10, "luas_ha")
    st.subheader(f"10 provinsi dengan karhutla terluas, {thn}")
    r = r.assign(kelas_hist=r["luas_ha"].map(lambda v: kelas_dari_luas(v, AMBANG)))
    top = alt.Chart(r).mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("luas_ha:Q", title="Luas (ha)"), y=alt.Y("provinsi:N", sort="-x", title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("kelas_hist:N", scale=alt.Scale(domain=KELAS, range=[HEX[k] for k in KELAS]),
                        legend=alt.Legend(title="Kelas")),
        tooltip=["provinsi", alt.Tooltip("luas_ha:Q", format=",.0f"), "kelas_hist"])
    st.altair_chart(top.properties(height=320), width="stretch")


def halaman_tentang():
    hero("Tentang", "Sumber data, cara model dibuat, dan keterbatasan yang perlu diketahui.")
    st.markdown(f"""
**Tujuan.** Mengklasifikasikan kelas kerawanan karhutla provinsi (rendah/sedang/tinggi) memakai Random Forest.

**Data.**
- Luas karhutla per provinsi 2018-2026: SiPongi (KLHK/Kemenhut).
- Curah hujan bulanan: NASA POWER (satu titik ibu kota per provinsi), diolah menjadi total kemarau Jul-Okt.
- Indeks Kualitas Ekosistem Gambut 2024, Indeks Kualitas Udara 2024 (rata-rata kabupaten/kota per provinsi),
  dan Indeks Perilaku Ramah Lingkungan Hidup 2025.

**Label.** Kelas dari luas terbakar tahun itu: rendah < {AMBANG[0]} ha, sedang {AMBANG[0]}-{AMBANG[1]} ha, tinggi > {AMBANG[1]} ha.

**Evaluasi.** Latih 2019-2023, uji 2024-2025, dibandingkan dengan baseline dan divalidasi silang antar provinsi.

**Keterbatasan.**
- Sampel kecil (38 provinsi x 7 tahun), sehingga hasil uji belum tentu stabil.
- Model setara baseline "sama dengan tahun lalu"; peningkatannya tidak signifikan secara statistik.
- Curah hujan hanya satu titik per provinsi dan tidak mewakili wilayah yang luas.
- Gambut, IKU, dan IPRLH bersifat statis dan hampir tidak berpengaruh pada model.
- Provinsi tanpa data gambut diberi nilai 0 dengan penanda has_gambut = 0 (asumsi).
- Data Papua sebelum 2023 tidak dipakai karena batas provinsi berubah akibat pemekaran.
- Bukan prediksi setahun ke depan: curah hujan tahun berjalan dipakai sebagai masukan.
- Label berupa luas terbakar per provinsi, bukan peta kerawanan spasial.
""")

HALAMAN = {
    "Prediksi": halaman_prediksi,
    "Performa Model": halaman_performa,
    "Data Historis": halaman_data,
    "Tentang": halaman_tentang,
}
pilih = st.radio("Halaman", list(HALAMAN), horizontal=True, label_visibility="collapsed")
HALAMAN[pilih]()