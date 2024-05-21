<h1 align="center">Analyse des Sentiments dans la Gestion des Changements en Entreprise avec les Modèles Zero-Shot</h1>

---

## Objectif du Projet
Ce projet vise à automatiser la classification des réactions des employés face aux changements organisationnels en exploitant des algorithmes de traitement du langage naturel. Notre application d'analyse des sentiments cherche à identifier le degré d'engagement des employés vis-à-vis des changements proposés.

## Contexte et Justification
Dans le cadre de la gestion des changements en entreprise, l'analyse des communications écrites et orales des employés est essentielle. Cependant, l'indisponibilité des données étiquetées nous a orientés vers l'utilisation de modèles de classification zero-shot. Ces modèles, qui ne nécessitent pas de données étiquetées pour des classes spécifiques, permettent une grande flexibilité et adaptabilité à des tâches non vues auparavant.

## Solution Proposée
Nous utilisons Les grands modèles de langage (LLM), disponibles en open source sur la plateforme Hugging Face, et les adaptons à notre tâche via des techniques de prompting. Une évaluation comparative de ces modèles a été réalisée pour identifier le plus performant pour notre application spécifique.

## Fichiers et Répertoires
- **Models_Zero_Shot_Evaluation.csv** : Contient les résultats d'évaluation des modèles sur plusieurs benchmarks.
- **Gestion des changements.csv** : Dataset d'évaluation comprenant des extraits variés des communications des employés.
- **Models_SA.ipynb** : Notebook Jupyter pour l'évaluation comparative des modèles.
- **streamlit_App/App.py** : Application Streamlit développée pour la visualisation interactive des résultats.
- **streamlit_App/plot_utils.py** : Contient des outils de plotting et d'autres fonctions utilisées dans l'application Streamlit.
- **requirements.txt** : Liste des bibliothèques nécessaires pour exécuter l'application.

## Déploiement de l'Application
Pour exécuter l'application Streamlit localement, suivez ces étapes :
1. Clonez le dépôt sur votre machine locale.
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
3. Naviguez vers le dossier streamlit_App et lancez l'application :
   ```bash
   streamlit run App.py
