"""
Pipeline kerawanan karhutla per provinsi (Random Forest).

Jalankan dari folder KarhutlaAI:
    python src/pipeline.py

Input  (taruh di data/raw/):
    luas-karhutla-provinsi.xlsx
    Indeks_Kualitas_Ekosistem_Gambut__2024_.csv
    Indeks_Kualitas_Udara__2024_.csv
    Indeks_Perilaku_Ramah_Lingkungan_Hidup__2025_.csv
    curah_hujan_tahunan.csv

Output:
    data/processed/dataset_panel.csv
    model/model_rf.joblib, model/meta.json
    hasil/*.csv, hasil/*.png, hasil/ringkasan.txt

Pertanyaan yang dijawab model:
    "Diberi riwayat karhutla provinsi dan curah hujan musim kemarau tahun t,
     termasuk kelas kerawanan apa provinsi itu di tahun t?"
Catatan: curah hujan tahun t dipakai sebagai input (bukan ramalan setahun ke depan).
"""
import io
import json
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold

# ----------------------------------------------------------------- konfigurasi
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("KARHUTLA_RAW", os.path.join(BASE, "data", "raw"))
PROC = os.path.join(BASE, "data", "processed")
HASIL = os.path.join(BASE, "hasil")
MODEL = os.path.join(BASE, "model")
for d in (PROC, HASIL, MODEL):
    os.makedirs(d, exist_ok=True)

SEED = 42
TAHUN_AWAL, TAHUN_AKHIR = 2018, 2025          # 2026 belum setahun penuh -> tidak dipakai
TRAIN_SAMPAI = 2023                            # latih 2019-2023, uji 2024-2025
REF_SAMPAI = 2023                              # periode acuan anomali hujan (hindari bocor)
AMBANG = (1000, 10000)                         # ha: <1000 rendah | 1000-10000 sedang | >10000 tinggi
KELAS = ["rendah", "sedang", "tinggi"]
# provinsi yang batas wilayahnya berubah (pemekaran Papua 2022): data sebelum 2023 tidak dipakai
BATAS_BERUBAH = {
    "Papua": 2023, "Papua Barat Daya": 2023, "Papua Pegunungan": 2023,
    "Papua Selatan": 2023, "Papua Tengah": 2023,
}

# penyeragaman nama provinsi antar sumber
ALIAS = {
    "Nanggroe Aceh Darussalam": "Aceh",
    "Kepulauan Bangka Belitung": "Bangka Belitung", "Kep. Bangka Belitung": "Bangka Belitung",
    "Kep Bangka Belitung": "Bangka Belitung",
    "Kep. Riau": "Kepulauan Riau", "Kep Riau": "Kepulauan Riau",
    "Daerah Istimewa Yogyakarta": "Yogyakarta", "DI Yogyakarta": "Yogyakarta",
    "NTB": "Nusa Tenggara Barat", "NTT": "Nusa Tenggara Timur",
}


def nama(s):
    s = str(s).strip()
    return ALIAS.get(s, s)


def angka_id(s):
    """'1.284,70' -> 1284.70"""
    return pd.to_numeric(
        s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def kelas_dari_luas(luas):
    return np.where(luas < AMBANG[0], 0, np.where(luas <= AMBANG[1], 1, 2))


# ------------------------------------------------------------------- muat data
def muat_luas():
    df = pd.read_excel(os.path.join(RAW, "luas-karhutla-provinsi.xlsx"), dtype=str)
    thn = [str(y) for y in range(2018, 2027)]
    df.columns = ["kode", "provinsi"] + thn
    df = df[df["provinsi"].str.strip().str.lower() != "total"].copy()
    df["provinsi"] = df["provinsi"].map(nama)
    for t in thn:
        df[t] = angka_id(df[t])
    long = df.melt(id_vars="provinsi", value_vars=thn, var_name="tahun", value_name="luas_ha")
    long["tahun"] = long["tahun"].astype(int)
    return long


def muat_hujan():
    h = pd.read_csv(os.path.join(RAW, "curah_hujan_tahunan.csv"))
    h["provinsi"] = h["provinsi"].map(nama)
    h = h[["provinsi", "tahun", "ch_tahunan_mm", "ch_kemarau_mm"]].copy()
    # anomali dihitung dari periode acuan saja (<= REF_SAMPAI) supaya data uji tidak bocor
    ref = h[h["tahun"] <= REF_SAMPAI].groupby("provinsi")["ch_kemarau_mm"].agg(["mean", "std"])
    h = h.merge(ref, left_on="provinsi", right_index=True)
    h["ch_kemarau_anomali_mm"] = h["ch_kemarau_mm"] - h["mean"]
    h["ch_kemarau_z"] = h["ch_kemarau_anomali_mm"] / h["std"].replace(0, np.nan)
    return h.drop(columns=["mean", "std"])


def muat_gambut():
    g = pd.read_csv(os.path.join(RAW, "Indeks_Kualitas_Ekosistem_Gambut__2024_.csv"), decimal=",")
    g.columns = ["kode", "provinsi", "gambut", "kategori"]
    g = g[g["provinsi"].str.upper() != "NASIONAL"].copy()
    g["provinsi"] = g["provinsi"].map(nama)
    return g[["provinsi", "gambut"]]


def muat_iku():
    u = pd.read_csv(os.path.join(RAW, "Indeks_Kualitas_Udara__2024_.csv"))
    u["provinsi"] = u["Provinsi"].map(nama)
    return u.groupby("provinsi")["IKU"].mean().rename("iku_prov").reset_index()


def muat_iprlh():
    raw = open(os.path.join(RAW, "Indeks_Perilaku_Ramah_Lingkungan_Hidup__2025_.csv"),
               encoding="utf-8").read()
    raw = raw.replace("Pulau/Kepulauan \ndan Provinsi", "Provinsi")  # header terpotong
    p = pd.read_csv(io.StringIO(raw), decimal=",")
    p["provinsi"] = p["Provinsi"].map(nama)
    return p[["provinsi", "IPRLH"]].rename(columns={"IPRLH": "iprlh"})


# ------------------------------------------------------------ bangun dataset
def bangun_panel():
    luas = muat_luas()
    luas = luas[(luas["tahun"] >= TAHUN_AWAL) & (luas["tahun"] <= TAHUN_AKHIR)].copy()
    luas["log_luas"] = np.log1p(luas["luas_ha"])
    luas["valid_dari"] = luas["provinsi"].map(BATAS_BERUBAH).fillna(TAHUN_AWAL).astype(int)
    luas["data_sah"] = luas["tahun"] >= luas["valid_dari"]
    luas = luas.sort_values(["provinsi", "tahun"]).reset_index(drop=True)

    # lag hanya dari tahun yang sah (batas wilayah sama)
    luas["log_sah"] = luas["log_luas"].where(luas["data_sah"])
    g = luas.groupby("provinsi")
    luas["lag1_log_luas"] = g["log_sah"].shift(1)
    luas["lag2_log_luas"] = g["log_sah"].shift(2)
    luas["mean_lag_log_luas"] = luas[["lag1_log_luas", "lag2_log_luas"]].mean(axis=1)

    df = luas.merge(muat_hujan(), on=["provinsi", "tahun"], how="left")
    df = df.merge(muat_gambut(), on="provinsi", how="left")
    df["has_gambut"] = df["gambut"].notna().astype(int)   # tanpa data gambut diasumsikan tidak ada gambut berarti
    df["gambut"] = df["gambut"].fillna(0)
    df = df.merge(muat_iku(), on="provinsi", how="left").merge(muat_iprlh(), on="provinsi", how="left")

    df["kelas"] = kelas_dari_luas(df["luas_ha"].values)
    df["kelas_lag1"] = np.where(df["lag1_log_luas"].notna(),
                                kelas_dari_luas(np.expm1(df["lag1_log_luas"]).values), np.nan)

    # baris layak: tahun sah, ada lag1, ada hujan
    df["layak"] = df["data_sah"] & df["lag1_log_luas"].notna() & df["ch_kemarau_mm"].notna()
    return df


FITUR_A = ["lag1_log_luas", "mean_lag_log_luas"]
FITUR_B = FITUR_A + ["ch_kemarau_mm", "ch_kemarau_z", "ch_tahunan_mm"]
FITUR_C = FITUR_B + ["gambut", "has_gambut", "iku_prov", "iprlh"]
SET_FITUR = {"A: riwayat saja": FITUR_A, "B: + curah hujan": FITUR_B, "C: + gambut, IKU, IPRLH": FITUR_C}


def buat_rf(**kw):
    p = dict(n_estimators=200, random_state=SEED, class_weight="balanced", n_jobs=-1)
    p.update(kw)
    return RandomForestClassifier(**p)


def tune(Xtr, ytr, grup):
    grid = {"max_depth": [3, 6, None], "min_samples_leaf": [1, 3, 5]}
    gs = GridSearchCV(buat_rf(n_estimators=100), grid, cv=GroupKFold(n_splits=5),
                      scoring="f1_macro", n_jobs=1)
    gs.fit(Xtr, ytr, groups=grup)
    return gs.best_params_, gs.best_score_


def metrik(y, p):
    return {
        "akurasi": accuracy_score(y, p),
        "balanced_acc": balanced_accuracy_score(y, p),
        "f1_macro": f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0),
    }


# ----------------------------------------------------------------------- main
def main():
    df = bangun_panel()
    layak = df[df["layak"]].copy()
    df.to_csv(os.path.join(PROC, "dataset_panel.csv"), index=False)

    train = layak[layak["tahun"] <= TRAIN_SAMPAI]
    test = layak[layak["tahun"] > TRAIN_SAMPAI]
    ytr, yte = train["kelas"].astype(int), test["kelas"].astype(int)
    out = []
    out.append("RINGKASAN HASIL")
    out.append("=" * 60)
    out.append(f"Baris layak: {len(layak)} | latih (2019-{TRAIN_SAMPAI}): {len(train)} | "
               f"uji ({TRAIN_SAMPAI+1}-{TAHUN_AKHIR}): {len(test)}")
    out.append(f"Provinsi: {layak['provinsi'].nunique()}")
    out.append(f"Ambang kelas (ha): rendah <{AMBANG[0]} | sedang {AMBANG[0]}-{AMBANG[1]} | tinggi >{AMBANG[1]}")
    dist = layak.groupby("tahun")["kelas"].value_counts().unstack(fill_value=0)
    dist.columns = [KELAS[int(c)] for c in dist.columns]
    out.append("\nSebaran kelas per tahun:\n" + dist.to_string())
    out.append("\nSebaran kelas latih: " + str(ytr.value_counts().sort_index().rename(dict(enumerate(KELAS))).to_dict()))
    out.append("Sebaran kelas uji  : " + str(yte.value_counts().sort_index().rename(dict(enumerate(KELAS))).to_dict()))

    # ---- baseline
    base = []
    mayor = ytr.mode()[0]
    base.append(("Baseline: kelas terbanyak", metrik(yte, np.full(len(yte), mayor))))
    base.append(("Baseline: sama dgn tahun lalu", metrik(yte, test["kelas_lag1"].astype(int))))

    # ---- RF tiap set fitur (uji temporal) + CV antar provinsi
    hasil, model_terbaik = [], {}
    for nm, fitur in SET_FITUR.items():
        bp, cv = tune(train[fitur], ytr, train["provinsi"])
        rf = buat_rf(**bp).fit(train[fitur], ytr)
        pred = rf.predict(test[fitur])
        m = metrik(yte, pred)
        hasil.append({"model": f"RF {nm}", **m, "f1_cv_antar_provinsi": cv, "param": str(bp)})
        model_terbaik[nm] = (rf, fitur, pred)
    for nm, m in base:
        hasil.append({"model": nm, **m, "f1_cv_antar_provinsi": np.nan, "param": ""})
    tabel = pd.DataFrame(hasil)
    tabel.to_csv(os.path.join(HASIL, "tabel_perbandingan_model.csv"), index=False)
    out.append("\nPERBANDINGAN MODEL (uji temporal 2024-2025)\n" +
               tabel.drop(columns="param").round(3).to_string(index=False))

    # ---- model utama = set C
    nm_c = "C: + gambut, IKU, IPRLH"
    rf, fitur, pred = model_terbaik[nm_c]

    # ---- apakah RF benar-benar lebih baik dari 'sama dengan tahun lalu'? (bootstrap data uji)
    rng = np.random.default_rng(SEED)
    yv, pers = yte.values, test["kelas_lag1"].astype(int).values
    n = len(yv)

    def f1m(a, b, i):
        return f1_score(a[i], b[i], average="macro", labels=[0, 1, 2], zero_division=0)

    out.append("\nSELISIH F1-macro vs baseline 'sama dengan tahun lalu' (bootstrap 2000x, 95% CI)")
    for nm, (_, _, pr) in model_terbaik.items():
        d = []
        for _ in range(2000):
            i = rng.integers(0, n, n)
            d.append(f1m(yv, pr, i) - f1m(yv, pers, i))
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = "signifikan" if (lo > 0 or hi < 0) else "TIDAK signifikan (CI memuat 0)"
        out.append(f"  RF {nm}: {np.mean(d):+.3f}  [{lo:+.3f}, {hi:+.3f}]  -> {sig}")
    berubah = pers != yv
    out.append(f"  Baris yang kelasnya berubah dari tahun lalu: {berubah.sum()} dari {n}; "
               f"akurasi RF C pada baris itu: {(pred[berubah] == yv[berubah]).mean():.2f} "
               f"(baseline = 0 karena selalu menebak kelas lama)")
    cm = confusion_matrix(yte, pred, labels=[0, 1, 2])
    pd.DataFrame(cm, index=[f"asli_{k}" for k in KELAS],
                 columns=[f"prediksi_{k}" for k in KELAS]).to_csv(os.path.join(HASIL, "confusion_matrix.csv"))
    out.append("\nLaporan klasifikasi model C:\n" +
               classification_report(yte, pred, labels=[0, 1, 2], target_names=KELAS, zero_division=0))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=KELAS).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (uji 2024-2025)")
    plt.tight_layout(); plt.savefig(os.path.join(HASIL, "confusion_matrix.png"), dpi=150); plt.close()

    # ---- feature importance
    imp = pd.DataFrame({"fitur": fitur, "gini": rf.feature_importances_})
    pi = permutation_importance(rf, test[fitur], yte, n_repeats=10, random_state=SEED,
                                scoring="f1_macro", n_jobs=-1)
    imp["permutasi"] = pi.importances_mean
    imp["permutasi_std"] = pi.importances_std
    imp = imp.sort_values("permutasi", ascending=True)
    imp.sort_values("permutasi", ascending=False).to_csv(os.path.join(HASIL, "feature_importance.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].barh(imp["fitur"], imp["gini"], color="#2b7a78"); ax[0].set_title("Importance (Gini)")
    ax[1].barh(imp["fitur"], imp["permutasi"], xerr=imp["permutasi_std"], color="#c06c3e")
    ax[1].set_title("Importance permutasi (turun F1-macro)")
    plt.tight_layout(); plt.savefig(os.path.join(HASIL, "feature_importance.png"), dpi=150); plt.close()
    out.append("Feature importance (urut permutasi):\n" +
               imp.sort_values("permutasi", ascending=False).round(4).to_string(index=False))

    # ---- grafik perbandingan model
    fig, ax = plt.subplots(figsize=(8, 4.2))
    t2 = tabel.copy(); t2["model"] = t2["model"].str.replace("RF ", "RF\n", regex=False)
    ax.bar(range(len(t2)), t2["f1_macro"], color=["#2b7a78"] * 3 + ["#9aa5a5"] * 2)
    ax.set_xticks(range(len(t2))); ax.set_xticklabels(t2["model"], fontsize=7)
    ax.set_ylabel("F1-macro (uji 2024-2025)"); ax.set_ylim(0, 1)
    plt.tight_layout(); plt.savefig(os.path.join(HASIL, "perbandingan_model.png"), dpi=150); plt.close()

    # ---- model final: latih di semua baris layak (2019-2025) untuk aplikasi
    bp_final, _ = tune(layak[fitur], layak["kelas"].astype(int), layak["provinsi"])
    final = buat_rf(**bp_final).fit(layak[fitur], layak["kelas"].astype(int))
    joblib.dump(final, os.path.join(MODEL, "model_rf.joblib"))
    ref = df[df["tahun"] <= REF_SAMPAI].groupby("provinsi")["ch_kemarau_mm"].agg(["mean", "std"])
    stat = df.drop_duplicates("provinsi").set_index("provinsi")[["gambut", "has_gambut", "iku_prov", "iprlh"]]
    terakhir = df[df["tahun"] == TAHUN_AKHIR].set_index("provinsi")[["luas_ha", "log_luas"]]
    prev = df[df["tahun"] == TAHUN_AKHIR - 1].set_index("provinsi")["log_sah"].rename("log_prev")
    meta = {
        "fitur": fitur, "kelas": KELAS, "ambang_ha": list(AMBANG), "param": {k: (None if v is None else v) for k, v in bp_final.items()},
        "latih": f"2019-{TAHUN_AKHIR}", "uji_temporal": f"{TRAIN_SAMPAI+1}-{TAHUN_AKHIR}",
        "f1_macro_uji": float(tabel.loc[tabel['model'].str.startswith('RF C'), 'f1_macro'].iloc[0]),
        "tahun_terakhir": TAHUN_AKHIR,
    }
    json.dump(meta, open(os.path.join(MODEL, "meta.json"), "w"), indent=2)
    profil = stat.join(ref.rename(columns={"mean": "ch_kemarau_ref_mean", "std": "ch_kemarau_ref_std"}))
    profil = profil.join(terakhir.rename(columns={"luas_ha": "luas_ha_terakhir", "log_luas": "log_luas_terakhir"}))
    profil = profil.join(prev)
    profil["ch_tahunan_ref_mean"] = df[df["tahun"] <= REF_SAMPAI].groupby("provinsi")["ch_tahunan_mm"].mean()
    profil.reset_index().to_csv(os.path.join(MODEL, "profil_provinsi.csv"), index=False)

    # ---- peringkat kerawanan tahun terakhir (data riil, untuk laporan)
    rk = layak[layak["tahun"] == TAHUN_AKHIR][["provinsi", "luas_ha", "kelas"]].copy()
    rk["kelas_asli"] = rk["kelas"].map(dict(enumerate(KELAS)))
    rk["kelas_prediksi"] = final.predict(layak[layak["tahun"] == TAHUN_AKHIR][fitur])
    rk["kelas_prediksi"] = rk["kelas_prediksi"].map(dict(enumerate(KELAS)))
    rk.drop(columns="kelas").sort_values("luas_ha", ascending=False).to_csv(
        os.path.join(HASIL, f"kelas_provinsi_{TAHUN_AKHIR}.csv"), index=False)

    out.append("\nCATATAN KETERBATASAN (salin ke laporan)")
    out.append("- Sampel kecil (<= 38 provinsi x 7 tahun); hasil bisa berubah bila periode uji berbeda.")
    out.append("- Curah hujan = satu titik ibu kota per provinsi (NASA POWER), tidak mewakili wilayah luas.")
    out.append("- Gambut/IKU (2024) dan IPRLH (2025) statis; dipakai sama untuk semua tahun.")
    out.append("- Provinsi tanpa data gambut diberi nilai 0 + penanda has_gambut=0 (asumsi).")
    out.append("- Data Papua sebelum 2023 tidak dipakai (batas provinsi berubah akibat pemekaran).")
    out.append("- Label = luas terbakar (SiPongi), dikelaskan dengan ambang tetap; bukan kerawanan spasial.")
    teks = "\n".join(out)
    open(os.path.join(HASIL, "ringkasan.txt"), "w", encoding="utf-8").write(teks)
    print(teks)


if __name__ == "__main__":
    main()