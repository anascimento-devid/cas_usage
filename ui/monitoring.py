from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Monitoring — Bank Marketing",
    layout="wide",
)

st.title("Monitoring — Bank Marketing")


METRICS_FILE = Path("monitoring/api_metrics.csv")
CAMPAIGN_FILE = Path("monitoring/campaign_results.csv")


# ============================================================
# Monitoring technique
# ============================================================

st.header("Santé du service")

if not METRICS_FILE.exists():
    st.warning("Aucune donnée technique disponible")

else:
    metrics = pd.read_csv(METRICS_FILE)

    metrics["timestamp"] = pd.to_datetime(
        metrics["timestamp"]
    )

    total_requests = len(metrics)

    successful_requests = (
        metrics["status_code"] < 400
    ).sum()

    availability = (
        successful_requests / total_requests
        if total_requests > 0
        else 0
    )

    error_rate = (
        metrics["status_code"] >= 400
    ).mean()

    latency_p95 = metrics[
        "latency_ms"
    ].quantile(0.95)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Requêtes",
        total_requests,
    )

    col2.metric(
        "Disponibilité",
        f"{availability:.1%}",
    )

    col3.metric(
        "Latence p95",
        f"{latency_p95:.1f} ms",
    )

    col4.metric(
        "Taux d'erreur",
        f"{error_rate:.1%}",
    )


    # ========================================================
    # Evolution de la latence
    # ========================================================

    st.subheader("Évolution de la latence")

    latency_history = (
        metrics
        .set_index("timestamp")[["latency_ms"]]
    )

    st.line_chart(latency_history)


    # ========================================================
    # Distribution des scores
    # ========================================================

    st.subheader("Distribution des scores")

    scores = metrics[
        "probability"
    ].dropna()

    if not scores.empty:
        bins = np.linspace(
            0,
            1,
            11,
        )

        score_distribution = (
            pd.cut(
                scores,
                bins=bins,
                include_lowest=True,
            )
            .value_counts(sort=False)
            .rename_axis("score_range")
            .reset_index(name="count")
        )

        score_distribution["score_range"] = (
            score_distribution[
                "score_range"
            ].astype(str)
        )

        st.bar_chart(
            score_distribution,
            x="score_range",
            y="count",
        )

    else:
        st.info(
            "Aucun score de souscription disponible"
        )


# ============================================================
# Monitoring métier
# ============================================================

st.header("Performance métier")

if not CAMPAIGN_FILE.exists():
    st.info(
        "Les métriques métier seront disponibles "
        "après récupération des résultats de campagne"
    )

else:
    campaign = pd.read_csv(
        CAMPAIGN_FILE
    )

    required_columns = {
        "probability",
        "y_true",
    }

    if not required_columns.issubset(
        campaign.columns
    ):
        st.error(
            "campaign_results.csv doit contenir "
            "les colonnes probability et y_true"
        )

    else:
        campaign = campaign.sort_values(
            "probability",
            ascending=False,
        )

        n_top = max(
            1,
            int(np.ceil(
                len(campaign) * 0.10
            )),
        )

        selected = campaign.head(
            n_top
        )

        precision_top10 = (
            selected["y_true"].mean()
        )

        total_subscribers = (
            campaign["y_true"].sum()
        )

        subscribers_captured = (
            selected["y_true"].sum()
            / total_subscribers
            if total_subscribers > 0
            else 0
        )

        baseline = (
            campaign["y_true"].mean()
        )

        lift = (
            precision_top10 / baseline
            if baseline > 0
            else 0
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Précision top 10 %",
            f"{precision_top10:.1%}",
        )

        col2.metric(
            "Souscripteurs captés",
            f"{subscribers_captured:.1%}",
        )

        col3.metric(
            "Lift",
            f"{lift:.2f}x",
        )


        # ====================================================
        # Calibration
        # ====================================================

        st.subheader("Calibration")

        mean_probability = (
            campaign[
                "probability"
            ].mean()
        )

        real_subscription_rate = (
            campaign[
                "y_true"
            ].mean()
        )

        calibration_gap = (
            real_subscription_rate
            - mean_probability
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Probabilité moyenne prédite",
            f"{mean_probability:.1%}",
        )

        col2.metric(
            "Souscription réelle",
            f"{real_subscription_rate:.1%}",
        )

        col3.metric(
            "Écart",
            f"{calibration_gap:+.1%}",
        )