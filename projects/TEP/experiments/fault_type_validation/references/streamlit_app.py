from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from torch import nn


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
METADATA_PATH = MODEL_DIR / "dl_model_metadata_pytorch.json"


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DEFAULT_TEST_PATH = first_existing_path(
    PROJECT_DIR / "dataset" / "TEP_Faulty_Testing.csv",
    PROJECT_DIR.parent / "dataset" / "TEP_Faulty_Testing.csv",
)
TUNER_DIR = PROJECT_DIR / "tuner"
PPT_GRAPH_DIR = PROJECT_DIR / "report" / "ppt_graphs"

PPT_GRAPH_FILES = {
    "forecaster": PPT_GRAPH_DIR / "01_forecaster_loss_actual_vs_predicted.png",
    "classifier": PPT_GRAPH_DIR / "02_classifier_accuracy_precision_recall_f1.png",
    "hyperparameter": PPT_GRAPH_DIR / "03_hyperparameter_search_comparison.png",
    "hyperparameter_top20": PPT_GRAPH_DIR / "04_hyperparameter_top20_summary.png",
}

EXPERIMENT_METRICS = pd.DataFrame(
    [
        ("1 actual + 1 predicted", "Accuracy", 0.7389),
        ("1 actual + 1 predicted", "Macro Precision", 0.8515),
        ("1 actual + 1 predicted", "Macro Recall", 0.8095),
        ("1 actual + 1 predicted", "Macro F1", 0.8182),
        ("9 actual + 1 predicted", "Accuracy", 0.7800),
        ("9 actual + 1 predicted", "Macro Precision", 0.8900),
        ("9 actual + 1 predicted", "Macro Recall", 0.8500),
        ("9 actual + 1 predicted", "Macro F1", 0.8600),
    ],
    columns=["experiment", "metric", "score"],
)

CLASS_REPORT = pd.DataFrame(
    [
        (0, 0.44, 0.25, 0.32, 1_500_000),
        (1, 1.00, 0.83, 0.91, 400_000),
        (2, 1.00, 0.83, 0.91, 400_000),
        (3, 0.59, 0.88, 0.71, 400_000),
        (4, 1.00, 0.98, 0.99, 400_000),
        (5, 1.00, 0.83, 0.91, 400_000),
        (6, 1.00, 0.83, 0.91, 400_000),
        (7, 1.00, 0.83, 0.91, 400_000),
        (8, 1.00, 0.89, 0.94, 400_000),
        (9, 0.52, 0.45, 0.48, 400_000),
        (10, 1.00, 0.96, 0.98, 400_000),
        (11, 0.98, 0.99, 0.98, 400_000),
        (12, 0.99, 0.91, 0.95, 400_000),
        (13, 1.00, 0.86, 0.92, 400_000),
        (14, 1.00, 0.99, 1.00, 400_000),
        (15, 0.22, 0.83, 0.35, 400_000),
        (16, 1.00, 0.97, 0.99, 400_000),
        (17, 1.00, 0.95, 0.97, 400_000),
        (18, 0.97, 0.87, 0.92, 400_000),
        (19, 1.00, 0.99, 1.00, 400_000),
        (20, 1.00, 0.93, 0.96, 400_000),
    ],
    columns=["faultNumber", "precision", "recall", "f1", "support"],
)

FORECASTER_HPARAMS = pd.DataFrame(
    [
        ("Model", "LSTM Forecaster"),
        ("Input window", "10 rows"),
        ("Input features", "52"),
        ("LSTM layers", "2"),
        ("LSTM units", "64, 64"),
        ("Dropout", "0.1"),
        ("Dense units", "192"),
        ("Loss", "Huber(delta=1.0)"),
        ("Optimizer", "AdamW"),
        ("Learning rate", "0.002089199259293805"),
    ],
    columns=["item", "value"],
)

CLASSIFIER_HPARAMS = pd.DataFrame(
    [
        ("Model", "Dense Classifier"),
        ("Input", "previous 9 actual rows + predicted next row"),
        ("Input dimension", "520"),
        ("Dense layers", "3"),
        ("Units", "448, 128, 448"),
        ("Dropout", "0.4, 0.1, 0.2"),
        ("BatchNorm", "after each hidden layer"),
        ("Loss", "CrossEntropyLoss with class weights"),
        ("Optimizer", "AdamW"),
        ("Learning rate", "0.0015372915308927418"),
    ],
    columns=["item", "value"],
)


class Forecaster(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.head = nn.Sequential(
            nn.Linear(64, 192),
            nn.ReLU(),
            nn.Linear(192, n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class FaultClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 448),
            nn.BatchNorm1d(448),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(448, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 448),
            nn.BatchNorm1d(448),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(448, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def read_metric_value(metrics: dict, metric_name: str):
    metric = metrics.get("metrics", {}).get(metric_name)
    if not metric:
        return None
    observations = metric.get("observations", [])
    if not observations:
        return None
    value = observations[-1].get("value", [None])
    return value[0] if value else None


def render_graph_image(title: str, path: Path, caption: str | None = None) -> None:
    st.markdown(f"**{title}**")
    if path.exists():
        st.image(str(path), use_container_width=True, caption=caption)
    else:
        st.info(
            f"Graph image not found: `{path}`\n\n"
            "Run the `PPT 발표자료용 그래프 생성` section at the bottom of the notebook to generate it."
        )


@st.cache_data(show_spinner=False)
def load_tuner_trials(project_name: str) -> pd.DataFrame:
    project_dir = TUNER_DIR / project_name
    rows = []
    if not project_dir.exists():
        return pd.DataFrame()

    for trial_path in project_dir.glob("trial_*/trial.json"):
        with open(trial_path, "r", encoding="utf-8") as f:
            trial = json.load(f)

        values = trial.get("hyperparameters", {}).get("values", {})
        metrics = trial.get("metrics", {})
        row = {
            "trial_id": trial.get("trial_id"),
            "status": trial.get("status"),
            "score": trial.get("score"),
            "best_step": trial.get("best_step"),
            "val_loss": read_metric_value(metrics, "val_loss"),
            "val_accuracy": read_metric_value(metrics, "val_accuracy"),
            "loss": read_metric_value(metrics, "loss"),
            "accuracy": read_metric_value(metrics, "accuracy"),
        }
        for key, value in values.items():
            if not key.startswith("tuner/"):
                row[key] = value
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df[df["status"] == "COMPLETED"].copy()


def resolve_artifact_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    # Metadata may contain Windows absolute paths such as
    # C:\work\dlproject\model\tep_feature_scaler.pkl. In the Docker
    # container those files are mounted under /app/model, so resolve by
    # filename as a portable fallback.
    raw_text = str(raw_path)
    windows_name = PureWindowsPath(raw_text).name
    if windows_name:
        fallback = MODEL_DIR / windows_name
        if fallback.exists():
            return fallback

    fallback = MODEL_DIR / path.name
    if fallback.exists():
        return fallback
    return path


@st.cache_resource
def load_artifacts():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_cols"]
    window_size = int(metadata["window_size"])

    scaler = joblib.load(resolve_artifact_path(metadata["scaler_path"]))
    label_encoder = joblib.load(resolve_artifact_path(metadata["label_encoder_path"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    forecaster = Forecaster(len(feature_cols)).to(device)
    forecaster.load_state_dict(
        torch.load(resolve_artifact_path(metadata["forecaster_path"]), map_location=device)
    )
    forecaster.eval()

    classifier = FaultClassifier(
        int(metadata.get("classifier_input_dim", len(feature_cols) * window_size)),
        len(label_encoder.classes_),
    ).to(device)
    classifier.load_state_dict(
        torch.load(resolve_artifact_path(metadata["classifier_path"]), map_location=device)
    )
    classifier.eval()

    return metadata, feature_cols, window_size, scaler, label_encoder, forecaster, classifier, device


@st.cache_data(show_spinner=False)
def load_default_data(max_rows: int) -> pd.DataFrame:
    if not DEFAULT_TEST_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(DEFAULT_TEST_PATH, nrows=max_rows)


@st.cache_data(show_spinner=False)
def load_balanced_fault_sample(feature_cols: list[str], max_per_fault: int = 3000) -> pd.DataFrame:
    if not DEFAULT_TEST_PATH.exists():
        return pd.DataFrame()

    usecols = feature_cols + ["faultNumber"]
    collected = {fault: [] for fault in range(21)}
    counts = {fault: 0 for fault in range(21)}

    for chunk in pd.read_csv(DEFAULT_TEST_PATH, usecols=usecols, chunksize=200_000):
        for fault in range(21):
            needed = max_per_fault - counts[fault]
            if needed <= 0:
                continue

            fault_rows = chunk[chunk["faultNumber"] == fault].head(needed)
            if not fault_rows.empty:
                collected[fault].append(fault_rows)
                counts[fault] += len(fault_rows)

        if all(count >= max_per_fault for count in counts.values()):
            break

    frames = [frame for fault_frames in collected.values() for frame in fault_frames]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def calculate_sensor_association(sample_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    if sample_df.empty or "faultNumber" not in sample_df.columns:
        return pd.DataFrame()

    feature_df = sample_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    global_mean = feature_df.mean()
    global_std = feature_df.std().replace(0, np.nan)

    rows = []
    for fault, group in sample_df.groupby("faultNumber"):
        class_mean = group[feature_cols].apply(pd.to_numeric, errors="coerce").mean()
        signed_effect = ((class_mean - global_mean) / global_std).replace([np.inf, -np.inf], np.nan).fillna(0)
        for sensor, value in signed_effect.items():
            rows.append(
                {
                    "faultNumber": int(fault),
                    "sensor": sensor,
                    "signed_effect": float(value),
                    "association": float(abs(value)),
                }
            )

    return pd.DataFrame(rows)


@torch.no_grad()
def predict_next_and_fault(recent_rows: pd.DataFrame):
    metadata, feature_cols, window_size, scaler, label_encoder, forecaster, classifier, device = load_artifacts()

    if len(recent_rows) != window_size:
        raise ValueError(f"recent_rows must contain exactly {window_size} rows")

    raw_window = recent_rows[feature_cols]
    scaled_window = scaler.transform(raw_window).astype(np.float32)

    seq = torch.from_numpy(scaled_window.reshape(1, window_size, len(feature_cols))).to(device)
    pred_next_scaled = forecaster(seq).detach().cpu().numpy()[0].astype(np.float32)
    pred_next_raw = scaler.inverse_transform(pred_next_scaled.reshape(1, -1))[0]

    prev_9_flat = scaled_window[1:].reshape(1, -1)
    classifier_input = np.concatenate([prev_9_flat, pred_next_scaled.reshape(1, -1)], axis=1).astype(np.float32)
    logits = classifier(torch.from_numpy(classifier_input).to(device))
    probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

    pred_next_df = pd.DataFrame([pred_next_raw], columns=feature_cols)
    proba_df = pd.DataFrame(
        {
            "faultNumber": label_encoder.classes_,
            "probability": probabilities,
        }
    ).sort_values("probability", ascending=False)

    return pred_next_df, proba_df


def render_title_summary() -> None:
    st.title("TEP 공정 이상 진단 프로젝트")

    st.markdown(
        "**프로젝트 개요:** Tennessee Eastman Process 데이터를 사용해 다음 시점의 공정 상태를 예측하고, "
        "예측 결과를 바탕으로 faultNumber를 분류하는 딥러닝 기반 이상 진단 프로젝트입니다."
    )
    st.markdown(
        "**선정된 모델:** LSTM Forecaster로 다음 시점 feature를 예측한 뒤, "
        "직전 9개 실제 row와 예측 row 1개를 결합해 Dense Classifier로 고장 유형을 분류합니다."
    )
    st.markdown(
        "**최종 성능:** 전체 테스트 기준 Accuracy 0.78, Macro Precision 0.89, "
        "Macro Recall 0.85, Macro F1 0.86을 기록했습니다."
    )


def render_performance_tab() -> None:
    st.subheader("Performance Metric Graphs")
    graph_cols = st.columns(2)
    with graph_cols[0]:
        render_graph_image(
            "Forecaster best loss + actual/predicted comparison",
            PPT_GRAPH_FILES["forecaster"],
            "Best validation loss trial and saved forecaster prediction quality",
        )
    with graph_cols[1]:
        render_graph_image(
            "Classifier Accuracy / Precision / Recall / F1",
            PPT_GRAPH_FILES["classifier"],
            "Saved classifier performance summary and class-wise F1",
        )

    st.subheader("Experiment Comparison")
    comparison = EXPERIMENT_METRICS.pivot(index="metric", columns="experiment", values="score")
    st.line_chart(comparison)

    st.subheader("Class F1")
    class_f1 = CLASS_REPORT.set_index("faultNumber")[["f1"]]
    st.bar_chart(class_f1)

    st.subheader("Class Precision / Recall")
    class_pr = CLASS_REPORT.set_index("faultNumber")[["precision", "recall"]]
    st.line_chart(class_pr)

    st.subheader("Class-Level Report")
    st.dataframe(
        CLASS_REPORT.style.format(
            {"precision": "{:.2f}", "recall": "{:.2f}", "f1": "{:.2f}", "support": "{:,}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    weak = CLASS_REPORT.sort_values("f1").head(4)
    st.info(
        "Weakest classes by F1: "
        + ", ".join(f"{int(row.faultNumber)} (F1 {row.f1:.2f})" for row in weak.itertuples())
    )


def render_hyperparameter_tab() -> None:
    st.subheader("Performance Metric Graphs")
    render_graph_image(
        "Hyperparameter Top 20 summary",
        PPT_GRAPH_FILES["hyperparameter_top20"],
        "Top 20 classifier trials sorted by validation accuracy",
    )
    with st.expander("Show More Performance Metric Graphs", expanded=False):
        render_graph_image(
            "Hyperparameter search comparison",
            PPT_GRAPH_FILES["hyperparameter"],
            "Trial ranking, learning-rate scatter, and dense-layer heatmap",
        )

    st.subheader("Final Hyperparameters")
    left, right = st.columns(2)
    with left:
        st.markdown("**Forecaster**")
        st.dataframe(FORECASTER_HPARAMS, use_container_width=True, hide_index=True)
    with right:
        st.markdown("**Classifier**")
        st.dataframe(CLASSIFIER_HPARAMS, use_container_width=True, hide_index=True)

    st.subheader("Performance By Classifier Input")
    comparison = EXPERIMENT_METRICS.pivot(index="metric", columns="experiment", values="score")
    st.line_chart(comparison)

    st.subheader("Experiment Table")
    table = comparison.reset_index()
    st.dataframe(table.style.format(precision=4), use_container_width=True, hide_index=True)

    st.subheader("KerasTuner Trial Results")
    trial_kind = st.radio(
        "Tuner project",
        ["Forecaster", "Classifier"],
        horizontal=True,
    )

    if trial_kind == "Forecaster":
        trials = load_tuner_trials("next_forecaster")
        score_col = "val_loss"
        ascending = True
        preferred_params = ["rnn_type", "rnn_layers", "rnn_units_0", "rnn_units_1", "dropout_0", "dropout_1", "dense_units", "learning_rate"]
    else:
        trials = load_tuner_trials("fault_classifier")
        score_col = "val_accuracy"
        ascending = False
        preferred_params = ["dense_layers", "units_0", "units_1", "units_2", "dropout_0", "dropout_1", "dropout_2", "learning_rate"]

    if trials.empty or score_col not in trials:
        st.warning("No tuner trial history was found.")
        return

    trials = trials.dropna(subset=[score_col]).copy()
    top_trials = trials.sort_values(score_col, ascending=ascending).head(20)
    top_chart = top_trials[["trial_id", score_col]].set_index("trial_id")
    st.markdown("**Top 20 Trials**")
    st.bar_chart(top_chart)

    numeric_params = [
        col for col in preferred_params
        if col in trials.columns and pd.api.types.is_numeric_dtype(trials[col])
    ]
    categorical_params = [
        col for col in preferred_params
        if col in trials.columns and not pd.api.types.is_numeric_dtype(trials[col])
    ]

    chart_cols = st.columns(2)
    with chart_cols[0]:
        if "learning_rate" in numeric_params:
            st.markdown("**Learning Rate vs Score**")
            lr_df = trials[["learning_rate", score_col]].dropna()
            st.scatter_chart(lr_df, x="learning_rate", y=score_col)

    with chart_cols[1]:
        selectable = categorical_params + numeric_params
        if selectable:
            selected_param = st.selectbox("Average score by hyperparameter", selectable)
            grouped = (
                trials[[selected_param, score_col]]
                .dropna()
                .groupby(selected_param, as_index=True)[score_col]
                .mean()
                .sort_index()
            )
            st.bar_chart(grouped)

    visible_cols = ["trial_id", score_col] + [p for p in preferred_params if p in trials.columns]
    st.markdown("**Trial Table**")
    st.dataframe(
        trials.sort_values(score_col, ascending=ascending)[visible_cols].head(30),
        use_container_width=True,
        hide_index=True,
    )


def render_sensor_association_tab(feature_cols: list[str]) -> None:
    st.subheader("FaultNumber별 센서 연관성")
    st.caption(
        "각 faultNumber에서 센서 평균이 전체 평균보다 얼마나 달라지는지 표준화한 값입니다. "
        "절댓값이 클수록 해당 오류와의 연관성이 크게 나타납니다."
    )

    sample_df = load_balanced_fault_sample(feature_cols)
    association = calculate_sensor_association(sample_df, feature_cols)
    if association.empty:
        st.warning("센서 연관성을 계산할 데이터가 없습니다.")
        return

    available_faults = sorted(association["faultNumber"].unique().tolist())
    selected_fault = st.selectbox("faultNumber 선택", available_faults)
    top_n = st.slider("상위 센서 개수", 5, 25, 12)

    selected = (
        association[association["faultNumber"] == selected_fault]
        .sort_values("association", ascending=False)
        .head(top_n)
        .copy()
    )

    st.markdown("**선택한 faultNumber의 주요 센서**")
    chart_df = selected.set_index("sensor")[["signed_effect"]]
    st.bar_chart(chart_df)

    st.markdown("**상위 센서 상세값**")
    st.dataframe(
        selected[["sensor", "signed_effect", "association"]].style.format(
            {"signed_effect": "{:.3f}", "association": "{:.3f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**전체 faultNumber - 전체 센서 연관성 히트맵**")
    heatmap = (
        association
        .pivot(index="faultNumber", columns="sensor", values="signed_effect")
        .reindex(columns=feature_cols)
        .sort_index()
    )
    st.dataframe(
        heatmap.style.format("{:.2f}").background_gradient(cmap="coolwarm", axis=None),
        use_container_width=True,
    )


def render_prediction_tab(df: pd.DataFrame, window_size: int) -> None:
    if df.empty:
        st.warning(f"Default test data not found: {DEFAULT_TEST_PATH}")
        return

    st.subheader("Test Data Preview")
    st.dataframe(df.head(20), use_container_width=True)

    if len(df) < window_size:
        st.error(f"At least {window_size} rows are required.")
        return

    max_start = len(df) - window_size
    col_a, col_b = st.columns([2, 1])
    start_idx = col_a.slider("Window start row", 0, max_start, 0)
    col_b.metric("Rows Loaded", f"{len(df):,}")

    recent_rows = df.iloc[start_idx:start_idx + window_size]

    if st.button("Predict Fault", type="primary"):
        pred_next_df, proba_df = predict_next_and_fault(recent_rows)
        top = proba_df.iloc[0]

        st.subheader("Prediction Result")
        cols = st.columns(2)
        cols[0].metric("Predicted faultNumber", int(top["faultNumber"]))
        cols[1].metric("Top Probability", f"{top['probability']:.4f}")

        chart_data = proba_df.head(10).copy()
        chart_data["faultNumber"] = chart_data["faultNumber"].astype(str)
        st.bar_chart(chart_data.set_index("faultNumber")["probability"])

        left, right = st.columns(2)
        left.subheader("Top Fault Probabilities")
        left.dataframe(proba_df.head(10), use_container_width=True, hide_index=True)

        right.subheader("Predicted Next Feature Snapshot")
        right.dataframe(pred_next_df.iloc[:, :12], use_container_width=True, hide_index=True)


def render_ppt_graphs_tab() -> None:
    st.subheader("Performance Metric Graphs")
    st.caption("Performance metric graphs")

    render_graph_image(
        "1. Forecaster best loss + actual/predicted comparison",
        PPT_GRAPH_FILES["forecaster"],
    )
    render_graph_image(
        "2. Classifier Accuracy / Precision / Recall / F1",
        PPT_GRAPH_FILES["classifier"],
    )
    render_graph_image(
        "3. Hyperparameter search comparison",
        PPT_GRAPH_FILES["hyperparameter"],
    )
    render_graph_image(
        "4. Hyperparameter Top 20 summary",
        PPT_GRAPH_FILES["hyperparameter_top20"],
    )


def main() -> None:
    st.set_page_config(page_title="TEP Fault Predictor", layout="wide")

    metadata, feature_cols, window_size, _, _, _, _, _ = load_artifacts()

    with st.sidebar:
        st.header("Data")
        max_rows = st.number_input(
            "Rows to load from test CSV",
            min_value=1_000,
            max_value=500_000,
            value=20_000,
            step=1_000,
        )
        st.caption(str(DEFAULT_TEST_PATH))

    df = load_default_data(int(max_rows))

    render_title_summary()

    tab_predict, tab_performance, tab_association, tab_hparams, tab_ppt_graphs, tab_metadata = st.tabs(
        ["Predict", "Performance", "Sensor Association", "Hyperparameters", "Graphs", "Metadata"]
    )
    with tab_predict:
        render_prediction_tab(df, window_size)
    with tab_performance:
        render_performance_tab()
    with tab_association:
        render_sensor_association_tab(feature_cols)
    with tab_hparams:
        render_hyperparameter_tab()
    with tab_ppt_graphs:
        render_ppt_graphs_tab()
    with tab_metadata:
        st.json(metadata)


if __name__ == "__main__":
    main()
