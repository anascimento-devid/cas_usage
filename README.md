# Bank Marketing ML

Projet de classification visant à estimer la probabilité de souscription à un dépôt à terme afin de prioriser les clients à contacter dans une campagne de marketing bancaire

Le projet couvre l'analyse des données, la modélisation, l'évaluation métier, l'analyse d'équité et l'industrialisation du modèle

## Modèle retenu

Le modèle final est un **Gradient Boosting** entraîné sur le scénario **S3**

Ce scénario exclut

- `duration`, indisponible avant l'appel
- `age`
- `job`
- `marital`
- `education`

Le modèle produit un score de probabilité utilisé pour classer les clients

Une politique **top 10 %** est étudiée comme hypothèse de capacité de campagne

Sur le jeu de test

- 55 % de taux de souscription dans le top 10 %
- 48,8 % des souscripteurs captés
- lift ≈ 4,88

## Stack

- Python
- Pandas
- Scikit-learn
- FastAPI
- Pydantic
- Streamlit
- Pytest
- Docker
- Docker Compose
- GitHub Actions

## Structure

```text
.
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── __init__.py
├── notebooks/
│   └── rendu_certif.ipynb
├── models/
│   └── pipeline.joblib
├── ui/
│   ├── app.py
│   └── monitoring.py
├── tests/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Installation

```bash
python -m venv .venv
pip install -r requirements.txt
```

Sous Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

## Lancer le notebook

```bash
jupyter notebook
```

Puis ouvrir

```text
notebooks/rendu_certif.ipynb
```

## Lancer l'API

```bash
uvicorn api.main:app --reload
```

Swagger

```text
http://localhost:8000/docs
```

Routes principales

- `GET /health`
- `POST /predict`

L'API retourne un score de souscription et ne décide pas directement quels clients doivent être ciblés

## Lancer l'interface

```bash
streamlit run ui/app.py
```

L'interface permet de tester le scoring sur un profil client

## Tests

```bash
python -m pytest -v
```

Les tests couvrent notamment

- `/health`
- `/predict`
- validation des entrées
- rejet des catégories invalides

## Docker

```bash
docker compose up --build
```

## Monitoring proposé

Le suivi en production prévoit notamment

- latence p95
- taux d'erreur API
- disponibilité
- drift des données via PSI
- précision top-k
- lift
- souscripteurs captés
- calibration
- taux de sélection par sous-groupe
- FPR et FNR par sous-groupe

Un tableau de bord Streamlit peut être utilisé pour rendre ces indicateurs accessibles à un utilisateur non technique

## Limites

- le top 10 % est une hypothèse de travail et non une contrainte métier définitive
- le dashboard de monitoring complet n'est pas implémenté
- le réentraînement automatique n'est pas implémenté
- les écarts entre sous-groupes doivent être surveillés en production
