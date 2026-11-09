from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits


# Kolom yang ingin diprediksi oleh model.
TARGET = "final_grade_pct"
# Baris dengan missing pada fitur atau target dikeluarkan; fitur angka distandardisasi.
NUMERIC_FEATURES = [
    "age",
    "grade_level",
    "family_size",
    "num_siblings",
    "single_parent_household",
    "annual_family_income_inr",
    "has_internet_access",
    "has_computer_or_laptop",
    "distance_to_school_km",
    "extra_tutoring",
    "parental_involvement_score",
    "study_hours_per_week",
    "attendance_rate_pct",
    "previous_year_grade_pct",
]
# Fitur kategori akan diubah menjadi kolom numerik dengan one-hot encoding.
CATEGORICAL_FEATURES = [
    "gender",
    "school_type",
    "residence_type",
    "family_income_bracket",
    "father_education_level",
    "mother_education_level",
    "father_occupation",
    "mother_occupation",
    "scholarship_status",
]
# Kolom ini dicatat dalam laporan, tetapi tidak dipakai sebagai prediktor.
EXCLUDED_COLUMNS = [
    "student_id",
    "survey_date",
    "academic_year",
    "grade_class",
    "at_risk_flag",
]

# Rentang valid berdasarkan definisi kolom dan skala pada dataset.
# Nilai di luar rentang ini dianggap tidak valid dan barisnya dikeluarkan.
VALID_RANGES = {
    "age": (5, 25),
    "grade_level": (1, 12),
    "family_size": (1, None),
    "num_siblings": (0, None),
    "single_parent_household": (0, 1),
    "annual_family_income_inr": (0, None),
    "has_internet_access": (0, 1),
    "has_computer_or_laptop": (0, 1),
    "distance_to_school_km": (0, None),
    "extra_tutoring": (0, 1),
    "parental_involvement_score": (1, 5),
    "study_hours_per_week": (0, None),
    "attendance_rate_pct": (0, 100),
    "previous_year_grade_pct": (0, 100),
    TARGET: (0, 100),
}


def find_iqr_outliers(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Laporkan nilai di luar 1.5 IQR per kolom."""
    records: list[dict[str, object]] = []
    for column in columns:
        values = data[column].dropna()
        if values.empty:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = data[column].notna() & ((data[column] < lower) | (data[column] > upper))
        records.extend(
            {
                "baris_csv": int(index),
                "kolom": column,
                "nilai": float(data.at[index, column]),
                "batas_bawah_iqr": float(lower),
                "batas_atas_iqr": float(upper),
            }
            for index in data.index[mask]
        )
    return pd.DataFrame(
        records,
        columns=["baris_csv", "kolom", "nilai", "batas_bawah_iqr", "batas_atas_iqr"],
    )


def iqr_bounds(values: pd.Series) -> tuple[float, float]:
    """Hitung batas outlier 1.5 IQR dari nilai non-kosong."""
    clean_values = values.dropna()
    q1, q3 = clean_values.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def build_preprocessing(cap_bounds: dict[str, tuple[float, float]]) -> ColumnTransformer:
    """Bangun preprocessing baru untuk tiap estimator agar fit tetap independen."""
    numeric_transformers = []
    for column in NUMERIC_FEATURES:
        steps = []
        if column in cap_bounds:
            lower, upper = cap_bounds[column]
            steps.append(
                ("iqr_cap", FunctionTransformer(np.clip, kw_args={"a_min": lower, "a_max": upper}))
            )
        steps.append(("scaler", StandardScaler()))
        numeric_transformers.append(
            (f"numeric_{column}", Pipeline(steps=steps), [column])
        )

    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    return ColumnTransformer(
        transformers=numeric_transformers
        + [("categorical", categorical_pipeline, CATEGORICAL_FEATURES)],
        sparse_threshold=0,
    )


def parse_args() -> argparse.Namespace:
    # Atur argumen command line dan nilai default untuk file masukan/keluaran.
    parser = argparse.ArgumentParser(
        description="Buat visualisasi dan model regresi nilai akhir siswa."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("student_grade_ses_dataset.csv"),
        help="Lokasi file CSV (default: student_grade_ses_dataset.csv di folder ini).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("analysis_output"),
        help="Folder untuk menyimpan grafik, metrik, prediksi, dan model.",
    )
    return parser.parse_args()


def save_visualizations(
    data: pd.DataFrame,
    actual: pd.Series,
    predicted: np.ndarray,
    output_dir: Path,
) -> None:
    # Buat grafik ringkasan data dan diagnostik hasil prediksi.
    sns.set_theme(style="whitegrid", palette="deep")

    # Histogram menunjukkan sebaran nilai akhir seluruh siswa.
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data[TARGET], bins=20, kde=True, ax=ax, color="#3572A5")
    ax.set(title="Distribusi Nilai Akhir Siswa", xlabel="Nilai akhir (%)", ylabel="Jumlah siswa")
    fig.tight_layout()
    fig.savefig(output_dir / "distribusi_nilai_akhir.png", dpi=160)
    plt.close(fig)

    # Scatter plot dan garis tren membandingkan nilai sebelumnya dengan nilai akhir.
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x="previous_year_grade_pct",
        y=TARGET,
        alpha=0.35,
        s=22,
        ax=ax,
    )
    sns.regplot(
        data=data,
        x="previous_year_grade_pct",
        y=TARGET,
        scatter=False,
        ci=None,
        color="#D1495B",
        ax=ax,
    )
    ax.set(
        title="Nilai Tahun Sebelumnya vs Nilai Akhir",
        xlabel="Nilai tahun sebelumnya (%)",
        ylabel="Nilai akhir (%)",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "nilai_sebelumnya_vs_akhir.png", dpi=160)
    plt.close(fig)

    # Hitung rata-rata nilai akhir per jenis sekolah untuk grafik batang.
    school_means = (
        data.groupby("school_type", dropna=False)[TARGET]
        .mean()
        .sort_values(ascending=False)
        .rename("Rata-rata nilai akhir")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=school_means,
        x="school_type",
        y="Rata-rata nilai akhir",
        hue="school_type",
        legend=False,
        ax=ax,
    )
    ax.set(
        title="Rata-rata Nilai Akhir Menurut Jenis Sekolah",
        xlabel="Jenis sekolah",
        ylabel="Rata-rata nilai akhir (%)",
    )
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "rata_rata_jenis_sekolah.png", dpi=160)
    plt.close(fig)

    # Grafik diagnostik: kecocokan aktual-prediksi dan sebaran kesalahan (residual).
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(actual, predicted, alpha=0.45, s=22, color="#3572A5")
    low = min(float(actual.min()), float(predicted.min()))
    high = max(float(actual.max()), float(predicted.max()))
    axes[0].plot([low, high], [low, high], linestyle="--", color="#D1495B")
    axes[0].set(
        title="Nilai Aktual vs Prediksi (Data Uji)",
        xlabel="Nilai aktual (%)",
        ylabel="Nilai prediksi (%)",
    )
    residuals = actual.to_numpy() - predicted
    sns.histplot(residuals, bins=25, kde=True, ax=axes[1], color="#4C956C")
    axes[1].axvline(0, linestyle="--", color="#D1495B")
    axes[1].set(
        title="Distribusi Residual (Aktual - Prediksi)",
        xlabel="Residual (poin nilai)",
        ylabel="Jumlah siswa",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "diagnostik_regresi.png", dpi=160)
    plt.close(fig)


def main() -> None:
    # Baca argumen, pastikan CSV tersedia, lalu muat data ke DataFrame.
    args = parse_args()
    if not args.csv_path.is_file():
        raise FileNotFoundError(f"File CSV tidak ditemukan: {args.csv_path}")

    data = pd.read_csv(args.csv_path)
    # Pastikan semua kolom yang dibutuhkan analisis tersedia dalam file.
    required_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    required_columns = [TARGET, *required_features, "school_type"]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Kolom wajib tidak ditemukan di CSV: {missing_columns}")

    # Ubah kolom angka ke tipe numerik; nilai yang tidak bisa dikonversi menjadi kosong.
    numeric_columns = NUMERIC_FEATURES + [TARGET]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Buang baris yang missing pada target atau fitur yang digunakan model.
    model_columns = [TARGET, *required_features]
    missing_values_before_drop = {
        column: int(data[column].isna().sum()) for column in required_features
    }
    missing_target_values_before_drop = int(data[TARGET].isna().sum())
    missing_rows_mask = data[model_columns].isna().any(axis=1)
    missing_rows_removed = int(missing_rows_mask.sum())
    data = data.loc[~missing_rows_mask].copy()

    # Hapus baris dengan nilai numerik di luar rentang yang didefinisikan.
    invalid_by_column: dict[str, int] = {}
    valid_rows = pd.Series(True, index=data.index)
    for column, (lower, upper) in VALID_RANGES.items():
        invalid = pd.Series(False, index=data.index)
        if lower is not None:
            invalid |= data[column].notna() & (data[column] < lower)
        if upper is not None:
            invalid |= data[column].notna() & (data[column] > upper)
        invalid_by_column[column] = int(invalid.sum())
        valid_rows &= ~invalid
    invalid_range_rows = int((~valid_rows).sum())
    data = data.loc[valid_rows].copy()
    if len(data) < 2:
        raise ValueError("Diperlukan minimal dua baris dengan nilai target yang valid.")

    # Catat outlier pada data yang lolos validasi rentang.
    outliers = find_iqr_outliers(data, numeric_columns)
    handling_labels = {
        "annual_family_income_inr": "winsorize IQR pada fitur model",
        "distance_to_school_km": "winsorize IQR pada fitur model",
        "parental_involvement_score": "hapus baris outlier dari data latih saja",
        "final_grade_pct": "batas logis 0-100; outlier IQR yang valid dipertahankan",
        "previous_year_grade_pct": "batas logis 0-100; outlier IQR yang valid dipertahankan",
        "attendance_rate_pct": "batas logis 0-100; outlier IQR yang valid dipertahankan",
        "study_hours_per_week": "tandai saja; nilai dipertahankan",
        "num_siblings": "tandai saja; nilai dipertahankan",
        "family_size": "tandai saja; nilai dipertahankan",
    }
    if not outliers.empty:
        outliers["penanganan"] = outliers["kolom"].map(handling_labels)

    features = data[required_features].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        data[TARGET],
        test_size=0.2,
        random_state=45,
    )

    # Hapus outlier parental_involvement_score hanya dari data latih.
    parent_lower, parent_upper = iqr_bounds(X_train["parental_involvement_score"])
    parent_outlier_mask = X_train["parental_involvement_score"].notna() & (
        (X_train["parental_involvement_score"] < parent_lower)
        | (X_train["parental_involvement_score"] > parent_upper)
    )
    parental_outlier_train_rows_removed = int(parent_outlier_mask.sum())
    X_train = X_train.loc[~parent_outlier_mask].copy()
    y_train = y_train.loc[~parent_outlier_mask].copy()

    # Batas winsorization dipelajari dari data latih saja agar data uji tidak bocor.
    cap_bounds = {
        column: iqr_bounds(X_train[column])
        for column in ["annual_family_income_inr", "distance_to_school_km"]
    }
    capped_value_counts = {
        column: int((
            X_train[column].notna()
            & ((X_train[column] < lower) | (X_train[column] > upper))
        ).sum())
        for column, (lower, upper) in cap_bounds.items()
    }

    # Bandingkan lima estimator dengan pembagian data dan preprocessing yang sama.
    model_specs = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=20_000),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=1
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
    }
    fitted_models = {}
    predictions_by_model = {}
    comparison_rows = []
    for model_name, estimator in model_specs.items():
        candidate = Pipeline(
            steps=[
                ("preprocessing", build_preprocessing(cap_bounds)),
                ("regression", estimator),
            ]
        )
        # Batasi native thread pool agar eksekusi stabil pada environment terbatas.
        with threadpool_limits(limits=1):
            candidate.fit(X_train, y_train)
            candidate_predictions = candidate.predict(X_test)
        fitted_models[model_name] = candidate
        predictions_by_model[model_name] = candidate_predictions
        comparison_rows.append(
            {
                "model": model_name,
                "r2": float(r2_score(y_test, candidate_predictions)),
                "mae": float(mean_absolute_error(y_test, candidate_predictions)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, candidate_predictions))),
            }
        )

    comparison_rows.sort(key=lambda row: row["mae"])
    comparison = pd.DataFrame(comparison_rows)
    comparison.insert(0, "peringkat_mae", range(1, len(comparison) + 1))
    best_model_name = comparison_rows[0]["model"]
    best_model = fitted_models[best_model_name]
    predictions = predictions_by_model[best_model_name]

    # Baseline menebak nilai rata-rata data latih sebagai pembanding sederhana.
    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    # Simpan metrik kualitas model, statistik target, dan jumlah data kosong.
    metrics = {
        "model": best_model_name,
        "best_model": best_model_name,
        "model_comparison": comparison_rows,
        "target": TARGET,
        "rows_used": int(len(data) - parental_outlier_train_rows_removed),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42,
        "r2": comparison_rows[0]["r2"],
        "mae": comparison_rows[0]["mae"],
        "rmse": comparison_rows[0]["rmse"],
        "baseline_mean_mae": float(mean_absolute_error(y_test, baseline_predictions)),
        "target_mean": float(data[TARGET].mean()),
        "target_std": float(data[TARGET].std()),
        "missing_values_by_feature_before_drop": missing_values_before_drop,
        "missing_target_values_before_drop": missing_target_values_before_drop,
        "missing_rows_removed": missing_rows_removed,
        "invalid_rows_removed": invalid_range_rows,
        "invalid_values_by_column": invalid_by_column,
        "iqr_outlier_rule": "1.5 * IQR; feature-specific policies in outlier_handling",
        "iqr_outlier_values_flagged": int(len(outliers)),
        "iqr_outlier_counts_by_column": {
            column: int(count)
            for column, count in outliers["kolom"].value_counts().items()
        },
        "outlier_handling": {
            "annual_family_income_inr": {
                "policy": "IQR winsorization using training-set bounds",
                "lower_bound": cap_bounds["annual_family_income_inr"][0],
                "upper_bound": cap_bounds["annual_family_income_inr"][1],
                "training_values_capped": capped_value_counts["annual_family_income_inr"],
            },
            "distance_to_school_km": {
                "policy": "IQR winsorization using training-set bounds",
                "lower_bound": cap_bounds["distance_to_school_km"][0],
                "upper_bound": cap_bounds["distance_to_school_km"][1],
                "training_values_capped": capped_value_counts["distance_to_school_km"],
            },
            "parental_involvement_score": {
                "policy": "remove IQR outlier rows from training set only",
                "training_rows_removed": parental_outlier_train_rows_removed,
                "lower_bound": parent_lower,
                "upper_bound": parent_upper,
            },
            "final_grade_pct": {
                "policy": "constrain to logical range 0-100",
                "values_changed": int(((data[TARGET] < 0) | (data[TARGET] > 100)).sum()),
                "note": "IQR outliers within 0-100 remain unchanged",
            },
            "previous_year_grade_pct": {
                "policy": "constrain to logical range 0-100",
                "values_changed": int(((data["previous_year_grade_pct"] < 0) | (data["previous_year_grade_pct"] > 100)).sum()),
                "note": "IQR outliers within 0-100 remain unchanged",
            },
            "attendance_rate_pct": {
                "policy": "constrain to logical range 0-100",
                "values_changed": int(((data["attendance_rate_pct"] < 0) | (data["attendance_rate_pct"] > 100)).sum()),
                "note": "IQR outliers within 0-100 remain unchanged",
            },
            "study_hours_per_week": {"policy": "flag only; retain values"},
            "num_siblings": {"policy": "flag only; retain values"},
            "family_size": {"policy": "flag only; retain values"},
        },
        "excluded_columns": EXCLUDED_COLUMNS,
    }
    # Nilai positif menunjukkan MAE model lebih rendah daripada baseline.
    baseline_improvement_pct = (
        (metrics["baseline_mean_mae"] - metrics["mae"])
        / metrics["baseline_mean_mae"]
        * 100
    )

    # Buat folder keluaran dan simpan grafik, metrik, prediksi, serta model terlatih.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_visualizations(data, y_test, predictions, args.output_dir)
    outliers.to_csv(args.output_dir / "outlier_iqr.csv", index=False)
    comparison.to_csv(args.output_dir / "perbandingan_model.csv", index=False)
    (args.output_dir / "metrik_model.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    prediction_output = pd.DataFrame(
        {"nilai_aktual": y_test.to_numpy()}, index=y_test.index
    )
    for model_name, model_predictions in predictions_by_model.items():
        prediction_output[f"prediksi_{model_name}"] = model_predictions
        prediction_output[f"residual_{model_name}"] = y_test.to_numpy() - model_predictions
    prediction_output.to_csv(
        args.output_dir / "prediksi_data_uji.csv", index_label="baris_csv"
    )
    joblib.dump(fitted_models, args.output_dir / "model_regresi_semua.joblib")
    joblib.dump(best_model, args.output_dir / "model_regresi_terbaik.joblib")
    joblib.dump(
        fitted_models["LinearRegression"],
        args.output_dir / "model_regresi_linear.joblib",
    )
    # Susun laporan teks ringkas untuk dibaca tanpa membuka file metrik JSON.
    comparison_report = "\n".join(
        f"{rank}. {row['model']}: MAE {row['mae']:.2f}, "
        f"RMSE {row['rmse']:.2f}, R2 {row['r2']:.3f}"
        for rank, row in enumerate(comparison_rows, start=1)
    )
    report = (
        "RINGKASAN ANALISIS NILAI SISWA\n"
        "================================\n"
        f"Jumlah data: {metrics['rows_used']:,} siswa\n"
        f"Baris dihapus karena nilai di luar rentang valid: {invalid_range_rows}\n"
        f"Baris latih dihapus karena outlier parental_involvement_score: "
        f"{parental_outlier_train_rows_removed}\n"
        f"Nilai pendapatan/jarak yang dibatasi dengan IQR pada data latih: "
        f"{sum(capped_value_counts.values())}\n"
        f"Baris dihapus karena missing pada fitur/target: {missing_rows_removed}\n"
        f"Nilai outlier IQR yang terdeteksi: {len(outliers)}\n"
        f"Target: {TARGET}\n"
        f"Model terbaik berdasarkan MAE: {best_model_name}\n"
        "Pembagian data: 80% latih dan 20% uji (random_state=42)\n\n"
        "PERBANDINGAN MODEL PADA DATA UJI (diurutkan berdasarkan MAE)\n"
        f"{comparison_report}\n\n"
        "HASIL MODEL TERBAIK\n"
        f"R2: {metrics['r2']:.3f} "
        "(proporsi variasi nilai yang dijelaskan model)\n"
        f"MAE: {metrics['mae']:.2f} poin nilai\n"
        f"RMSE: {metrics['rmse']:.2f} poin nilai\n"
        f"MAE baseline tebakan rata-rata: {metrics['baseline_mean_mae']:.2f} poin nilai\n"
        f"Penurunan MAE dibanding baseline: {baseline_improvement_pct:.1f}%\n\n"
        "Catatan: grade_class dan at_risk_flag tidak digunakan sebagai prediktor "
        "untuk menghindari kebocoran informasi dari target. Baris dengan missing "
        "pada fitur atau target dibuang sebelum pemodelan; "
        "kebijakan outlier "
        "per kolom tercatat di metrik_model.json dan nilai yang terdeteksi tercatat "
        "di outlier_iqr.csv. Persentase yang sudah berada di 0-100 tidak berubah "
        "oleh pembatasan rentang logis. Hasil ini menunjukkan hubungan prediktif, "
        "bukan bukti sebab-akibat.\n"
    )
    (args.output_dir / "ringkasan_analisis.txt").write_text(report, encoding="utf-8")

    print(f"Analisis selesai. Hasil tersimpan di: {args.output_dir.resolve()}")
    print(f"Jumlah baris digunakan: {metrics['rows_used']}")
    print(f"R2: {metrics['r2']:.3f}")
    print(f"MAE: {metrics['mae']:.2f} poin nilai")
    print(f"RMSE: {metrics['rmse']:.2f} poin nilai")
    print(f"Model terbaik (MAE): {best_model_name}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    # Jalankan analisis hanya ketika file dieksekusi langsung.
    main()
