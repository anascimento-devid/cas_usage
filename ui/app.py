import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="📞",
)

st.title("Score de souscription")
st.write(
    "Estimation de la probabilité de souscription d'un client. "
    "Le score individuel sert ensuite à prioriser les clients "
    "dans le cadre d'une campagne."
)

default = st.selectbox(
    "Défaut de paiement",
    ["no", "yes", "unknown"],
)

housing = st.selectbox(
    "Crédit immobilier",
    ["no", "yes", "unknown"],
)

loan = st.selectbox(
    "Prêt personnel",
    ["no", "yes", "unknown"],
)

contact = st.selectbox(
    "Canal de contact",
    ["cellular", "telephone"],
)

month = st.selectbox(
    "Mois",
    [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ],
)

day_of_week = st.selectbox(
    "Jour",
    ["mon", "tue", "wed", "thu", "fri"],
)

campaign = st.number_input(
    "Nombre de contacts pendant la campagne",
    min_value=1,
    value=1,
)

pdays = st.number_input(
    "Jours depuis le dernier contact",
    min_value=0,
    value=999,
)

previous = st.number_input(
    "Nombre de contacts précédents",
    min_value=0,
    value=0,
)

poutcome = st.selectbox(
    "Résultat de la campagne précédente",
    ["nonexistent", "failure", "success"],
)

emp_var_rate = st.number_input(
    "Taux de variation de l'emploi",
    value=0.0,
)

cons_price_idx = st.number_input(
    "Indice des prix à la consommation",
    value=93.0,
)

cons_conf_idx = st.number_input(
    "Indice de confiance des consommateurs",
    value=-40.0,
)

euribor3m = st.number_input(
    "Euribor 3 mois",
    value=4.0,
)

nr_employed = st.number_input(
    "Nombre d'employés",
    value=5000.0,
)

if st.button("Scorer le client"):
    payload = {
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed,
    }

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=5,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        st.error(f"Impossible de contacter l'API : {exc}")
    else:
        result = response.json()
        probability = result["probability"]

        st.metric(
            "Score estimé de souscription",
            f"{probability:.1%}",
        )

        st.info(
            "Ce score est utilisé pour classer les clients d'une campagne. "
            "La décision de ciblage dépend ensuite de la capacité disponible "
            "et du classement de l'ensemble des clients."
        )

        if "model_version" in result:
            st.caption(
                f"Version du modèle : {result['model_version']}"
            )