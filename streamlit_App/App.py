import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import create_carte_allies, create_bar_chart, create_heatmap, matrice_confusion, evaluation_1, evaluation_2 , evaluation_3, assign_group, evolution_f1_score



avis = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Gestion des changements.csv")
df2 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/Models_Zero_Shot_Evaluation.csv")
# Nettoyage

avis.dropna(inplace=True)
avis["synergy"] = avis["synergy"].replace({"Interessé": "Intéressé", "Engagé ": "Engagé"})
avis["antagonism"] = avis["antagonism"].replace({"Concillant": "Conciliant"})
avis["class"] = avis.apply(assign_group, axis=1)
avis0 =avis
data11 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_bart_large_mnli_1.csv")
data21 = pd.read_csv(/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_deberta_v3_large_zeroshot_v2_0_1.csv")
data31 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_deberta_v3_large_zeroshot_v2_0_c_1.csv")
data41 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_deberta_v3_base_zeroshot_v2_0_1.csv")
data51 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_roberta_large_zeroshot_v2_0_1.csv")
data61 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_bge_m3_zeroshot_v2_0_1.csv")
data12 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_bart_large_mnli_2.csv")
data22 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_deberta_v3_large_zeroshot_v2_0_2.csv")
data32 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_deberta_v3_large_zeroshot_v2_0_c_2.csv")
data42 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_deberta_v3_base_zeroshot_v2_0_2.csv")
data52 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_roberta_large_zeroshot_v2_0_2.csv")
data62 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_bge_m3_zeroshot_v2_0_2.csv")
data13 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_bart_large_mnli_3.csv")
data23 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_deberta_v3_large_zeroshot_v2_0_3.csv")
data33 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_deberta_v3_large_zeroshot_v2_0_c_3.csv")
data43 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_deberta_v3_base_zeroshot_v2_0_3.csv")
data53 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_roberta_large_zeroshot_v2_0_3.csv")
data63 = pd.read_csv("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/Resultats/predictions_bge_m3_zeroshot_v2_0_3.csv")
def data(model,prompt):
        if model == "facebook/bart-large-mnli":
            if prompt == "prompt_1":
                return data11
            elif prompt =="prompt_2":
                return data12
            else :
                return data13
        elif model =="MoritzLaurer/deberta-v3-large-zeroshot-v2.0":
            if prompt == "prompt_1":
                return data21
            elif prompt =="prompt_2":
                return data22
            else :
                return data23
        elif model =="MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c":
            if prompt == "prompt_1":
                return data31
            elif prompt =="prompt_2":
                return data32
            else :
                return data33
        elif model =="MoritzLaurer/deberta-v3-base-zeroshot-v2.0":
            if prompt == "prompt_1":
                return data41
            elif prompt =="prompt_2":
                return data42
            else :
                return data43
        elif model =="MoritzLaurer/roberta-large-zeroshot-v2.0":
            if prompt == "prompt_1":
                return data51
            elif prompt =="prompt_2":
                return data52
            else :
                return data53
        elif model =="MoritzLaurer/bge-m3-zeroshot-v2.0":
            if prompt == "prompt_1":
                return data61
            elif prompt =="prompt_2":
                return data62
            else :
                return data63

st.sidebar.title("Sommaire")

pages = ["Contexte du projet", "Zero-shot-Learning","Préparation des données","Modélisation","Résultats", "Conclusion"]
page = st.sidebar.radio("Allez vers la page", pages)
if page == pages[0]:
    st.write(" # Gestion du changement")
    st.write("")
    st.write("Dans le domaine du management, le changement est un processus indispensable permettant aux entreprises de s'adapter et de prospérer face à des défis internes ou externes, en modifiant leurs structures, processus, technologies ou cultures. La qualification du changement intervient tôt dans ce processus, évaluant la complexité et la portée du changement envisagé pour définir les stratégies et les ressources nécessaires. Cela inclut la cartographie des acteurs, qui identifie et visualise toutes les parties prenantes affectées par ou impliquées dans le changement. En particulier, la cartographie des alliés se révèle cruciale pour isoler et mobiliser ceux qui soutiennent activement le projet. Cette cartographie est développée en plusieurs étapes clés : d'abord, l'identification de tous les acteurs impliqués dans le projet, suivie par l'évaluation de leur niveau d'implication et leur perception du changement. Ensuite, ces acteurs sont classés en différentes catégories telles que les avocats, les relais, et les opposants, en fonction de leur degré de soutien. Cette segmentation permet de cibler efficacement les communications et les actions de gestion du changement, facilitant ainsi la surmontée de la résistance et l'engagement en faveur du changement, essentiels pour la réussite du projet de changement. Ces outils aident à comprendre les différents niveaux de soutien ou de résistance, essentiels pour la réussite du projet de changement. ")
    st.write("")
    st.write("La tâche de cartographie des alliès peut être grandement améliorée par l'automatisation, en utilisant des outils technologiques pour collecter et analyser des données sur l'implication des acteurs de manière efficace.  Cette approche permet de visualiser  les interactions et les niveaux de soutien, et de tenir ces informations à jour avec peu d'effort manuel. A cet égard, l'intégration de l'analyse des sentiments peut  enrichir véritablement ce processus, en apportant une compréhension approfondie de la perception du changement par les acteurs. \n En fait,  l'analyse des sentiments utilise le traitement du langage naturel pour évaluer les communications écrites et orales au sein de l'organisation. Cette technologie peut détecter les nuances émotionnelles dans les emails, les messages instantanés, les réunions et les autres formes de communication.  Cela permet des interventions plus ciblées et personnalisées. Les gestionnaires peuvent ajuster les stratégies de communication pour mieux répondre aux préoccupations des acteurs, renforcer le soutien là où il est fort, et aborder proactivement les points de résistance")
    st.write("")
    st.write("Dans ce contexte, notre projet de 'la mise en place d'une application d'analyse des sentiments' vise à trouver une solution permettant de   classifier les acteurs dans différentes catégories selon leur réaction au changement. En exploitant des algorithmes  de traitement du langage naturel, l'application analyse les communications écrites et orales pour détecter non seulement les sentiments généraux - positifs, négatifs, ou neutres - mais également pour évaluer le degré d'engagement des employés vis-à-vis du changement proposé.")
    st.image("images/unnamed.jpg")
if page == pages[1]:
    st.write("# Zero shot learning")
    st.write("")
    st.write("")
    st.write("  \n\n\n Le domaine du traitement du langage naturel est très dynamique, profitant de l'apprentissage à partir de vastes quantités de données non étiquetées sur Internet. L'utilisation de modèles non supervisés a permis de dépasser les benchmarks dans l'apprentissage supervisé. Avec le développement continu de nouvelles architectures de modèles et objectifs d'apprentissage, les normes de performance évoluent rapidement, particulièrement pour les tâches nécessitant beaucoup de données étiquetées.")
    st.image("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/images/image4.png")
    st.write(" ### Zero-shot-classification, C'est quoi ?")
    st.write("L'apprentissage zéro-coup (ZSL) se définit non par un algorithme spécifique mais par son approche d'apprentissage où le modèle ne reçoit aucun exemple étiqueté des classes qu'il doit prédire après entraînement. Par exemple, certains grands modèles de langage (LLMs) sont bien adaptés aux tâches ZSL, car ils sont pré-entraînés par apprentissage auto-supervisé sur un corpus massif de textes qui peut contenir des références incidentes ou des connaissances sur des classes de données non vues. Sans exemples étiquetés sur lesquels s'appuyer, les méthodes ZSL dépendent toutes de l'utilisation de telles connaissances auxiliaires pour faire des prédictions.")
    st.write(" ### Zero-shot learning et classification de texte")
    st.image("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/images/image5.png")
    st.write("La résolution des tâches de classification de texte avec l'apprentissage zéro-coup peut servir d'exemple de la manière d'appliquer l'extrapolation des concepts appris au-delà du régime d'entraînement. Une manière de faire cela est d'utiliser l'inférence en langage naturel (NLI) comme proposé par Yin et al. (2019). Il existe également d'autres approches telles que le calcul des distances entre les plongements de texte ou la formulation du problème comme un test à trous. \n \nDans le NLI, la tâche consiste à déterminer si une hypothèse est vraie (implication), fausse (contradiction) ou indéterminée (neutre) étant donné une prémisse. Yin et al. (2019) ont proposé d'utiliser de grands modèles de langage comme BERT, entraînés sur des ensembles de données NLI, et d'exploiter leurs capacités de compréhension du langage pour la classification de texte en zero-shot. Cela peut se faire en prenant le texte d'intérêt comme prémisse et en formulant une hypothèse pour chaque catégorie potentielle en utilisant un modèle d'hypothèse. Ensuite, nous laissons le modèle NLI prédire si la prémisse implique l'hypothèse. Enfin, la probabilité prédite d'implication peut être interprétée comme la probabilité de l'étiquette.")
    st.write(" ### Classification de texte en zéro-shot avec Hugging Face 🤗")
    st.write("La variété des modèles de classification zero-shot sur Hugging Face montre une adaptabilité de cette technologie à de nombreux domaines d'application, des spécificités linguistiques, et des besoins en performance pour des tâches précises. Cela inclut des adaptations pour différentes langues, des optimisations basées sur des architectures neuronales variées telles que BERT et DeBERTa, et des modèles spécialisés pour des performances accrues dans des contextes comme l'inférence de langage ou la vérification des faits. La plateforme sert également de centre d'expérimentation pour la communauté de recherche, encourageant le partage, la personnalisation et l'amélioration continue des modèles. Cette diversité reflète non seulement les progrès technologiques mais aussi un intérêt croissant pour des solutions d'intelligence artificielle personnalisées et efficaces, adaptées aux besoins spécifiques des utilisateurs et des industries variées.")
    st.image("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/images/image6.png")
    st.write(" ### Quand utiliser quel modèle ?")
    st.write("**deberta-v3-zeroshot vs. roberta-zeroshot :** deberta-v3 offre de meilleures performances que roberta, mais il est un peu plus lent.  En résumé : pour la précision, utilisez un modèle deberta-v3. Si la vitesse d'inférence en production est une préoccupation, vous pouvez envisager un modèle roberta . \n \n **Cas d'utilisation commerciaux :** les modèles avec'-c' dans le titre sont garantis d'être entraînés uniquement sur des données commercialement amicales. Les modèles sans '-c' ont été entraînés sur plus de données et sont plus performants, mais incluent des données avec des licences non commerciales.  Pour les utilisateurs ayant des exigences légales strictes, il est recommandé d'utiliser les modèles avec '-c' dans le titre. \n \n **Cas d'utilisation multilingues/non-anglophones :** utilisez bge-m3-zeroshot-v2.0 ou bge-m3-zeroshot-v2.0-c. Notez que les modèles multilingues sont moins performants que les modèles uniquement en anglais. Vous pouvez donc également traduire d'abord vos textes en anglais avec des bibliothèques comme EasyNMT, puis appliquer n'importe quel modèle uniquement en anglais aux données traduites. La traduction automatique facilite également la validation si votre équipe ne parle pas toutes les langues des données. \n \n **Fenêtre de contexte :** Les modèles bge-m3 peuvent traiter jusqu'à 8192 jetons. Les autres modèles peuvent traiter jusqu'à 512. Notez que des entrées de texte plus longues rendent le modèle plus lent et diminuent la performance, donc si vous travaillez uniquement avec des textes allant jusqu'à environ 400 mots / 1 page, utilisez par exemple un modèle deberta pour de meilleures performances.")
    st.write(" ### Métriques")
    st.write("Les modèles ont été évalués sur 28 tâches différentes de classification de texte avec la métrique f1_macro. Le principal point de référence est facebook/bart-large-mnli qui est, au moment de la rédaction (03.04.24), le classificateur 0-shot le plus utilisé et commercialement convivial.")
    st.image("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/images/image3.png")
    # Afficher le DataFrame sans index
    df2 = df2.set_index("Dataset")
    st.dataframe(df2.head(29))
if page == pages[2]:
    st.write("## Préparation des données")
    st.write("")
    st.write("L'objectif de ce projet est de sélectionner un modèle de classification de texte en zero-shot learning qui permettra d'automatiser la classification des employés selon leur niveau de synergie et d'antagonisme vis-à-vis d'un projet. Nous cherchons à classer les employés dans les catégories suivantes : les classes de synergie, comprenant minimaliste, intéressé, coopérant et engagé, qui mesurent l'intérêt des membres de l'équipe; et les classes d'antagonisme, incluant conciliant, résistant, opposant et irréconciliable, qui évaluent leur résistance. \n \n Pour évaluer la performance de ces modèles dans notre tâche spécifique, un jeu de données a été créé en utilisant GPT-4. Ce jeu simule divers scénarios tels que la participation à des réunions, les réponses à des formulaires, et les feedbacks lors des entretiens de projet. Ces données, générées par  intelligence artificielle à partir de prompts spécifiques et classées manuellement, ne capturent pas entièrement la complexité et le contexte des situations réelles. Cependant, elles fournissent une base pour comparer et évaluer approximativement les capacités des modèles  à travers plusieurs métriques, dans ce contexte spécifique.")
    st.write(" ## Exploration des données")
    st.write("")
    st.dataframe(avis0.head(100))
    st.write("  Dimensions du Dataframe :")
    st.write(avis0.shape)
    if st.checkbox("  Afficher les valeurs manquantes"):
        st.write(avis0.isna().sum())
    if st.checkbox("  Afficher les doublons"):
        st.write(avis0.duplicated().sum())
    if st.checkbox("  Afficher les valeurs uniques"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("Synergie: ")
            st.write(avis0["synergy"].unique())


        with col2:
            st.write("Antagonisme: ")
            st.write(avis0["antagonism"].unique())
    st.write("## Visulaisation des données")   

    st.write("")
    st.write("")
    fig = create_carte_allies(avis0["synergy"], avis0["antagonism"])
    st.plotly_chart(fig) 
    fig2 = create_bar_chart(avis0["synergy"],"synergy")   
    st.plotly_chart(fig2)
    fig3= create_bar_chart(avis0["antagonism"],"antagonism") 
    st.plotly_chart(fig3)
    fig4 = create_heatmap(avis0)
    st.plotly_chart(fig4)
if page == pages[3] :
    
    st.write("## Modélisation")
    st.write("Dans le cadre de notre projet, l'objectif est de sélectionner un modèle de classification de texte en zero-shot learning capable d'automatiser efficacement la classification des employés selon leur niveau de synergie et d'antagonisme vis-à-vis d'un projet spécifique. Pour atteindre cet objectif, nous avons choisi d'évaluer une gamme de modèles de Transformers aux caractéristiques diverses afin de déterminer le plus adapté à nos besoins spécifiques. Parmi les modèles sélectionnés figurent :  **facebook/bart-large** ,  utilisé comme référence en raison de sa popularité et de ses performances éprouvées. **deberta-large** et **deberta-large-c**, pour évaluer l'impact de l'utilisation de données d'entraînement commercialement amicales seulement sur les performances ; **deberta-base**, reconnu pour sa rapidité ; **roberta-large**, une alternative rapide à deberta ; et **bge-m3**, qui est particulièrement adapté pour des applications multilingues ou le traitement de textes longs jusqu'à 8000 jetons, offrant un avantage significatif par rapport aux modèles standard qui supportent généralement autour de 500 jetons.\n nous utiliserons également l'API Gemini, un modèle de langue très puissant, pour comparer ses performances avec celles des modèles sélectionnés.")
    st.write("Il est également essentiel de considérer l'ingénierie de prompting, un aspect crucial qui peut influencer significativement les performances des modèles de classification en zero-shot learning. Le 'prompting' fait référence à la méthode de formulation des requêtes ou des instructions envoyées au modèle. Cette technique est utilisée pour orienter le modèle afin qu'il applique ses connaissances à une tâche spécifique.  \n Dans notre projet, nous explorons comment différentes stratégies de prompting peuvent améliorer la capacité du modèle.")
    model_choisi = st.selectbox (label = "Modèle", options = ["facebook/bart-large-mnli", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0", "MoritzLaurer/roberta-large-zeroshot-v2.0","MoritzLaurer/bge-m3-zeroshot-v2.0", "Api_Gemini"])
    Prompt = st.selectbox(label = "Prompt", options = ["prompt_1", "prompt_2","prompt_3"])
    st.write("### Tache 1 : classification selon son niveau de synergie ")
    for key , value in evaluation_1(model_choisi, Prompt).items():
        st.write(f'{key}: {value}')
    cm = evaluation_1(model_choisi, Prompt)["Matrice_confusion"]
    fig5 = matrice_confusion(cm, 1)
    st.plotly_chart(fig5)
    synergy = data(model_choisi, Prompt)["synergy"]
    st.write("##### Réalité : ")
   
    fig31= create_bar_chart(avis0["synergy"],"synergy") 
    st.plotly_chart(fig31)

    st.write("##### Prédictions : ")
    fig8 = create_bar_chart(synergy,"synergy")  
    st.plotly_chart(fig8)
      

    st.write("### Tache 2 : classification selon son niveau d'antagonisme ")
    for key , value in evaluation_2(model_choisi, Prompt).items():
        st.write(f'{key}: {value}')
    cm = evaluation_2(model_choisi, Prompt)["Matrice_confusion"]
    fig6 = matrice_confusion(cm, 2)
    st.plotly_chart(fig6)
    antagonism = data(model_choisi, Prompt)["antagonism"]
    st.write("##### Réalité : ")
   
    fig3= create_bar_chart(avis0["antagonism"],"antagonism") 
    st.plotly_chart(fig3)


    st.write("##### Prédictions : ")
    fig9= create_bar_chart(antagonism,"antagonism") 
    st.plotly_chart(fig9)
    
    
    st.write("### Tache 3 : classification finale")
    for key , value in evaluation_3(model_choisi, Prompt).items():
        st.write(f'{key}: {value}')
    cm = evaluation_3(model_choisi, Prompt)["Matrice_confusion"]
    fig7 = matrice_confusion(cm, 3)
    st.plotly_chart(fig7)
    group = data(model_choisi, Prompt)["class"]
    st.write("##### Réalité : ")
   
    fig3= create_bar_chart(avis0["class"],"class")
    st.plotly_chart(fig3)
    st.write("##### Prédictions : ")

    fig10= create_bar_chart(group,"class") 
    st.plotly_chart(fig10)
if page == pages[4]:

    st.write("# Résultats")
    st.write("")
    st.write("")
    st.write("##### Comparaison des scores F1 de différents modèles")
    st.image("/mount/src/l-analyse-des-sentiments-dans-la-gestion-des-changements-en-entreprise-avec-les-llms/streamlit_App/images/comparaison.png")
    models =["facebook/bart-large-mnli", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0", "MoritzLaurer/roberta-large-zeroshot-v2.0","MoritzLaurer/bge-m3-zeroshot-v2.0"]

    model_choisi = st.selectbox (label = "Modèle", options = ["facebook/bart-large-mnli", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0", "MoritzLaurer/roberta-large-zeroshot-v2.0","MoritzLaurer/bge-m3-zeroshot-v2.0"])
    fig = evolution_f1_score(model_choisi)
    st.pyplot(fig)


    
