import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
def hex_to_rgba(hex_color, alpha):
    # Enlever le caractère '#' s'il existe
    hex_color = hex_color.lstrip('#')
    
    # Convertir les valeurs hex en décimal
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    
    # Retourner la chaîne formatée en RGBA
    return f"rgba({r}, {g}, {b}, {alpha})"
regions = {
        ("Conciliant","Minimaliste" ): {'x_range': (0, 1), 'y_range': (0, 1) , "region": "Passif"},
        ("Conciliant", "Intéressé"): {'x_range': (0, 1), 'y_range': (1, 2),"region":"Passif"},
        ("Résistant", "Minimaliste"): {'x_range': (1, 2), 'y_range': (0, 1), "region":"Passif"},
        ("Résistant", "Intéressé"): {'x_range': (1, 2), 'y_range': (1, 2), "region":"Passif"} ,
        ("Conciliant", "Coopérant"): {'x_range': (0, 1), 'y_range': (2, 3), "region":"Aligné"},
        ("Conciliant", "Engagé"): {'x_range': (0, 1), 'y_range': (3, 4),"region":"Engagé"} ,
        ("Résistant", "Coopérant"): {'x_range': (1, 2), 'y_range': (2, 3), "region":"Moteur"},
        ("Résistant", "Engagé"): {'x_range': (1, 2), 'y_range': (3, 4),"region":"Moteur"},
        ("Opposant", "Coopérant"): {'x_range': (2, 3), 'y_range': (2, 3),"region":"Moteur"},
        ("Opposant", "Engagé"): {'x_range': (2, 3), 'y_range': (3, 4),"region":"Moteur"} ,
        ("Opposant", "Minimaliste"): {'x_range': (2, 3), 'y_range': (0, 1), "region":"Opposant"},
        ("Opposant", "Intéressé"): {'x_range': (2, 3), 'y_range': (1, 2),"region":"Opposant"},
        ("Irréconciliant", "Minimaliste"): {'x_range': (3, 4), 'y_range': (0, 1), "region":"Irréductible"},
        ("Irréconciliant", "Intéressé"): {'x_range': (3, 4), 'y_range': (1, 2), "region":"Irréductible"},
        ("Irréconciliant", "Coopérant"): {'x_range': (3, 4), 'y_range': (2, 3), "region":"Déchiré"},
        ("Irréconciliant", "Engagé"): {'x_range': (3, 4), 'y_range': (3, 4),"region":"Déchiré"}
        }

    
        
def assign_group(row):
            
            tuple = (row['antagonism'],row['synergy'])
            return regions [tuple]["region"]

def create_carte_allies(synergy, antagonism):
        synergy = synergy.tolist()
        antagonism = antagonism.tolist()
        
        ellipses = {
        (1, 1): ('Passifs', '#A6BDC2', 1.5, 1.7),
        (2.5, 1): ('Opposants', '#8FC0C8', 0.7, 1.9),
        (3.5, 1): ('Irréductibles', '#8CC9D4', 0.7, 1.9),
        (0.5, 2.5): ('Alignés', '#C8E7ED', 0.7, 0.9),
        (0.5, 3.5): ('Engagés', '#A0CAD1', 0.7, 0.9),
        (2, 3): ('Moteurs', '#6A9BA4', 1.7, 1.2),
        (3.5, 3): ('Déchirés', '#69BECD', 0.7, 1.7),
        (2, 2): ('Hésitants', '#B6DDE4', 0.5, 0.7)
    }

        annotations =[]
        shapes = [  
        {'type': 'line', 'x0': 1, 'y0': 0, 'x1': 1, 'y1': 4,
         'line': {'color': 'gray', 'width': 0.5, 'dash': 'dash'}},
         {'type': 'line', 'x0': 0, 'y0': 0, 'x1': 4, 'y1': 4,
         'line': {'color': 'gray', 'width': 0.5, 'dash': 'dash'}},
        {'type': 'line', 'x0': 2, 'y0': 0, 'x1': 2, 'y1': 4,
         'line': {'color': 'orange', 'width': 1.2, 'dash': 'dash'}},
        {'type': 'line', 'x0': 3, 'y0': 0, 'x1': 3, 'y1': 4,
         'line': {'color': 'gray', 'width': 0.5, 'dash': 'dash'}},
        {'type': 'line', 'x0': 0, 'y0': 1, 'x1': 4, 'y1': 1,
         'line': {'color': 'gray', 'width': 0.5, 'dash': 'dash'}},
        {'type': 'line', 'x0': 0, 'y0': 2, 'x1': 4, 'y1': 2,
         'line': {'color': "yellow", 'width': 1.2, 'dash': 'dash'}},
        {'type': 'line', 'x0': 0, 'y0': 3, 'x1': 4, 'y1': 3,
         'line': {'color': 'gray', 'width': 0.5, 'dash': 'dash'}},
        
    ]
# Create ellipses and annotations
        for (x, y), (label, color, width, height) in ellipses.items():

            shapes.append({
        'type': 'circle',
        'xref': 'x',
        'yref': 'y',
        'x0': x - width / 2,
        'y0': y - height / 2,
        'x1': x + width / 2,
        'y1': y + height / 2,
        'line_color':hex_to_rgba(color, 0.5),
        'line_width': 0 ,
        'fillcolor': hex_to_rgba(color, 0.5),
    })
            
            annotations.append({
        'x': x,
        'y': y,
        'xref': "x",
        'yref': "y",
        'text': label,
        'showarrow': False,
        'font': {
            'family': "sans serif",
            'size': 17,
            'color': 'white'
        },
        'bgcolor': hex_to_rgba(color, 0),
       'bordercolor': hex_to_rgba(color, 0),
       'borderwidth': 0
    })

    
        layout = go.Layout(
        width=800,  # Set the width of the figure
        height=600,  # Set the height of the figure
        xaxis=dict(
        title='Antagonisme',
        range=[0, 4],  # Set the range from 0 to 4
        tickmode='array',
        tickvals=[0.5, 1.5, 2.5, 3.5],  # Position labels in the middle of intervals
        ticktext=['Conciliant', 'Résistant', 'Opposant', 'Irréconciliable'],  # Labels for the x-axis
        linecolor='black',  # Color of the axis line
        showgrid=True,  # Enable grid lines on the y-axis
        gridcolor='rgba(255,255,255,0)',  # Set grid line color
        linewidth=1,  # Thickness of the axis line
        mirror=True,  # Reflect tick labels and axis lines on opposite side
         # Grid lines at every integer point
    ),
    yaxis=dict(
        title='Synergie',
        range=[0, 4],  # Set the range from 0 to 4
        tickmode='array',
        tickvals=[0.5, 1.5, 2.5, 3.5],  # Position labels in the middle of intervals
        ticktext=['Minimaliste', 'Intéressé', 'Coopérant', 'Engagé'],  # Labels for the y-axis
        linecolor='black',  # Color of the axis line
        showgrid=True,  # Enable grid lines on the y-axis
        gridcolor='rgba(255,255,255,0)',  # Set grid line color
        linewidth=1,  # Thickness of the axis line
        mirror=True,  # Reflect tick labels and axis lines on opposite side
         # Grid lines at every integer point
    ),
    shapes= shapes,
    annotations = annotations,
    plot_bgcolor='white'
)

# Define the data points
        trace = go.Scatter(
    x=[1],  # X position of the point, adjust as needed
    y=[2],  # Y position of the point, adjust as needed
    mode='markers',
    marker=dict(
        color='red',
        size=10
    ),
    name='Points'
)

    # Create a figure with the layout and data
        fig = go.Figure( layout=layout)
     
        assert len(synergy) == len(antagonism), "Les listes doivent avoir la même longueur"

#    Créer des tuples de paires correspondantes
# Exemple de données
        data = {
            'synergy': synergy,
            'antagonism': antagonism
}

        df = pd.DataFrame(data)
        paires = zip(antagonism, synergy)

        num_points  = Counter(paires)

      # Appliquer la fonction pour créer une nouvelle colonne 'Groupe'
        df['Groupe'] = df.apply(assign_group, axis=1)
        # Compter les occurrences de chaque groupe
        group_counts = df['Groupe'].value_counts()
        # Generate and plot points for each sub-region
        for key, bounds in regions.items():
            # Generate random points within the specified bounds
            x_coords = np.random.uniform(low=bounds['x_range'][0], high=bounds['x_range'][1], size=num_points[key])
            y_coords = np.random.uniform(low=bounds['y_range'][0], high=bounds['y_range'][1], size=num_points[key])
            region = regions[key]["region"]
            occ = group_counts.get(region, 0)
            fig.add_trace(go.Scatter(
            x= x_coords ,
            y=  y_coords,
            mode='markers',
            marker=dict(size=7),
            text=f'{region}: {occ}',
            hoverinfo='text'
))
            fig.update_layout(
         title={
        'text': "Carte des Alliés",
        'y':0.9,
        'x':0.65,
        'xanchor': 'center',
        'yanchor': 'top'
             })
            fig.update_layout(
    showlegend=False  # Désactiver l'affichage de la légende
)
   
   
        return fig
   
def create_bar_chart(data, category):
    if isinstance(data, pd.Series):
        data = data.tolist()
    df = pd.DataFrame(data, columns=[category])
    counts = df[category].value_counts(normalize=True) * 100
    counts = counts.reset_index()
    counts.columns = [category, 'Pourcentage']
    if category == "synergy":
        color_discrete_map={'Minimaliste': '#E0EBDB', 'Intéressé': '#AFED8E', 'Coopérant': '#8DB3ED', "Engagé" : '#FFF386'}
    elif category == "antagonism" :
        color_discrete_map={'Conciliant': '#F6FFD0', 'Résistant': '#FFFFB7', 'Opposant': '#FFBC7E', "Irréconciliant" : '#FF7171'}
    else :
        color_discrete_map={'Passif': '#F6FFD0', 'Aligné': '#FFFFB7', 'Engagé': '#FFBC7E', "Moteur" : '#FF7171', "Déchiré": '#FF7171', 'Opposant': '#FFFFB7', 'Irréductible': '#FFBC7E'}
    # Création du graphique avec Plotly
    fig = px.bar(counts, x=category, y='Pourcentage', 
                 labels={"Pourcentage": "Pourcentage% "},
                 text='Pourcentage', 
                 color=category,  # Définir la colonne pour les couleurs
                 color_discrete_map= color_discrete_map )
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    fig.update_layout(
    title={
        'text': "Répartition des niveaux de " +category,
        'y':0.9,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'
    }
)

    return fig

def create_heatmap(data):
    
    order_synergie = ['Minimaliste', 'Intéressé', 'Coopérant', 'Engagé']
    order_antagonisme = ['Conciliant', 'Résistant', 'Opposant', 'Irréconciliant']

    data['synergy'] = pd.Categorical(data['synergy'], categories=order_synergie, ordered=True)
    data['antagonism'] = pd.Categorical(data['antagonism'], categories=order_antagonisme, ordered=True)
    total = len(data['antagonism'])
# Compter les occurrences de chaque combinaison
    heatmap_data = data.groupby(['synergy', 'antagonism']).size().unstack(fill_value=0).reindex(index=order_synergie, columns=order_antagonisme)
    heatmap_data_1 = (heatmap_data / total) * 100
# Créer une matrice de couleurs personnalisées
    color_matrix = pd.DataFrame([
    ["#f4cccc", "#FFBC7E", "#e06666", "#cc0000"],  # Minimaliste
    ["#fce5cd", "#f9cb9c", "#f6b26b", "#e69138"],  # Intéressé
    ["#fff2cc", "#ffe599", "#ffd966", "#f1c232"],  # Coopérant
    ["#d9ead3", "#b6d7a8", "#93c47d", "#6aa84f"]   # Engagé
], columns=order_antagonisme, index=order_synergie)

# Créer le heatmap avec les couleurs personnalisées pour chaque cellule
    fig = px.imshow(
    heatmap_data_1,
    labels=dict(x="Catégorie d'Antagonisme", y="Catégorie de Synergie", color="Pourcentage de Combinaisons "),
    x=order_antagonisme,
    y=order_synergie,
    text_auto=True,
    aspect="auto",
    color_continuous_scale=None # Pas de scale de couleur discrète
)

# Appliquer la matrice de couleurs à la heatmap
    fig.update_traces(zmin=0, zmax=heatmap_data_1.max().max(), hoverinfo="z+text", hovertemplate="Synergie: %{y}<br>Antagonisme: %{x}<br>Pourcentage: %{z}<extra></extra>", showscale=False)
    fig.update_traces(z=heatmap_data_1.to_numpy(), colorscale=color_matrix.stack().tolist())

# Configuration additionnelle
    fig.update_xaxes(side="top")
# Configuration additionnelle
    fig.update_layout(
    title={
        'text': "Distribution des Combinaisons de Synergie et d'Antagonisme",
        'y':0.1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title="Antagonisme",
    yaxis_title="Synergie"
)
    


    return fig
def matrice_confusion(cm, task):
    class_names1=["Minimaliste", "Intéressé", "Coopérant","Engagé"]
    class_names2= ["Conciliant", "Résistant", "Opposant","Irréconciliant"]
    class_names3 = ["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"]

    if task == 1:
        labels = class_names1
    elif task == 2:
        labels = class_names2
    elif task == 3:
        labels = class_names3

    # Création de la figure Plotly
    fig = ff.create_annotated_heatmap(z=cm, x=labels, y=labels, colorscale='Blues', showscale=True)
    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted labels', yaxis_title='True labels')

    return fig
    
#avis = pd.read_csv('../Gestion des changements.csv')
# Nettoyage
#avis.dropna(inplace=True)
#avis["synergy"] = avis["synergy"].replace({"Interessé": "Intéressé", "Engagé ": "Engagé"})
#avis["antagonism"] = avis["antagonism"].replace({"Concillant": "Conciliant"})
#avis0 =avis

data11 = pd.read_csv("Resultats/predictions_bart_large_mnli_1.csv")
data21 = pd.read_csv("../streamlit_App/predictions_deberta_v3_large_zeroshot_v2_0_1.csv")
data31 = pd.read_csv("Résultats/predictions_deberta_v3_large_zeroshot_v2_0_c_1.csv")
data41 = pd.read_csv("Résultats/predictions_deberta_v3_base_zeroshot_v2_0_1.csv")
data51 = pd.read_csv("Résultats/predictions_roberta_large_zeroshot_v2_0_1.csv")
data61 = pd.read_csv("Résultats/predictions_bge_m3_zeroshot_v2_0_1.csv")
data12 = pd.read_csv("Résultats/predictions_bart_large_mnli_2.csv")
data22 = pd.read_csv("Résultats/predictions_deberta_v3_large_zeroshot_v2_0_2.csv")
data32 = pd.read_csv("Résultats/predictions_deberta_v3_large_zeroshot_v2_0_c_2.csv")
data42 = pd.read_csv("Résultats/predictions_deberta_v3_base_zeroshot_v2_0_2.csv")
data52 = pd.read_csv("Résultats/predictions_roberta_large_zeroshot_v2_0_2.csv")
data62 = pd.read_csv("Résultats/predictions_bge_m3_zeroshot_v2_0_2.csv")
data13 = pd.read_csv("Résultats/predictions_bart_large_mnli_3.csv")
data23 = pd.read_csv("Résultats/predictions_deberta_v3_large_zeroshot_v2_0_3.csv")
data33 = pd.read_csv("Résultats/predictions_deberta_v3_large_zeroshot_v2_0_c_3.csv")
data43 = pd.read_csv("Résultats/predictions_deberta_v3_base_zeroshot_v2_0_3.csv")
data53 = pd.read_csv("Résultats/predictions_roberta_large_zeroshot_v2_0_3.csv")
data63 = pd.read_csv("Résultats/predictions_bge_m3_zeroshot_v2_0_3.csv")
y_test_synergy = avis0["synergy"]
y_test_antagonism = avis0["antagonism"]
y_test = avis0["class"]
def evaluation_1 (model_choisi, prompt ):
        metrics = {"Matrice_confusion": "", "Précision": "", "Rappel": "", "Score F1" :"", "Accuracy": ""}
        if model_choisi == "facebook/bart-large-mnli":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data11["synergy"], labels = ["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                metrics["Précision"] = precision_score(y_test_synergy, data11["synergy"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_synergy, data11["synergy"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_synergy, data11["synergy"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data11["synergy"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data12["synergy"], labels =["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                 metrics["Précision"] = precision_score(y_test_synergy, data12["synergy"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test_synergy, data12["synergy"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test_synergy, data12["synergy"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data12["synergy"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data13["synergy"], labels = ["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                metrics["Précision"] = precision_score(y_test_synergy, data13["synergy"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_synergy, data13["synergy"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_synergy, data13["synergy"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data13["synergy"])
        elif model_choisi == "MoritzLaurer/deberta-v3-large-zeroshot-v2.0":
             if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data21["synergy"], labels = ["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                metrics["Précision"] = precision_score(y_test_synergy, data21["synergy"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_synergy, data21["synergy"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_synergy, data21["synergy"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data21["synergy"])
             elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data22["synergy"], labels = ["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                 metrics["Précision"] = precision_score(y_test_synergy, data22["synergy"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test_synergy, data22["synergy"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test_synergy, data22["synergy"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data22["synergy"])
             elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data23["synergy"], labels = ["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                metrics["Précision"] = precision_score(y_test_synergy, data23["synergy"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_synergy, data23["synergy"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_synergy, data23["synergy"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data23["synergy"])
        elif model_choisi == "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data31["synergy"], labels = ["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                metrics["Précision"] = precision_score(y_test_synergy, data31["synergy"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_synergy, data31["synergy"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_synergy, data31["synergy"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data31["synergy"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data32["synergy"], labels = ["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                 metrics["Précision"] = precision_score(y_test_synergy, data32["synergy"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test_synergy, data32["synergy"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test_synergy, data32["synergy"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data32["synergy"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data33["synergy"], labels =["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                metrics["Précision"] = precision_score(y_test_synergy, data33["synergy"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_synergy, data33["synergy"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_synergy, data33["synergy"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data33["synergy"])
        elif model_choisi == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data41["synergy"], labels = ["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                metrics["Précision"] = precision_score(y_test_synergy, data41["synergy"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_synergy, data41["synergy"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_synergy, data41["synergy"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data41["synergy"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data42["synergy"], labels = ["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                 metrics["Précision"] = precision_score(y_test_synergy, data42["synergy"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test_synergy, data42["synergy"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test_synergy, data42["synergy"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data42["synergy"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data43["synergy"], labels = ["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                metrics["Précision"] = precision_score(y_test_synergy, data43["synergy"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_synergy, data43["synergy"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_synergy, data43["synergy"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data43["synergy"])
        elif model_choisi == "MoritzLaurer/roberta-large-zeroshot-v2.0":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data51["synergy"], labels =["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                metrics["Précision"] = precision_score(y_test_synergy, data51["synergy"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_synergy, data51["synergy"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_synergy, data51["synergy"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data51["synergy"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data52["synergy"], labels =["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                 metrics["Précision"] = precision_score(y_test_synergy, data52["synergy"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test_synergy, data52["synergy"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test_synergy, data52["synergy"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data52["synergy"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data53["synergy"], labels =["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                metrics["Précision"] = precision_score(y_test_synergy, data53["synergy"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_synergy, data53["synergy"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_synergy, data53["synergy"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data53["synergy"])
        elif model_choisi == "MoritzLaurer/bge-m3-zeroshot-v2.0":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data61["synergy"], labels =["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                metrics["Précision"] = precision_score(y_test_synergy, data61["synergy"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_synergy, data61["synergy"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_synergy, data61["synergy"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data61["synergy"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data62["synergy"], labels = ["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                 metrics["Précision"] = precision_score(y_test_synergy, data62["synergy"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test_synergy, data62["synergy"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test_synergy, data62["synergy"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data62["synergy"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_synergy, data63["synergy"], labels = ["Minimaliste", "Intéressé", "Coopérant","Engagé"])
                metrics["Précision"] = precision_score(y_test_synergy, data63["synergy"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_synergy, data63["synergy"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_synergy, data63["synergy"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_synergy, data63["synergy"])
        return metrics
def evaluation_2 (model_choisi, prompt ):
        metrics = {"Matrice_confusion": "", "Précision": "", "Rappel": "", "Score F1" :"", "Accuracy": ""}
        if model_choisi == "facebook/bart-large-mnli":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data11["antagonism"], labels =["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                metrics["Précision"] = precision_score(y_test_antagonism, data11["antagonism"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_antagonism, data11["antagonism"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_antagonism, data11["antagonism"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data11["antagonism"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data12["antagonism"], labels = ["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                 metrics["Précision"] = precision_score(y_test_antagonism, data12["antagonism"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test_antagonism, data12["antagonism"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test_antagonism, data12["antagonism"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data12["antagonism"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data13["antagonism"], labels =["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                metrics["Précision"] = precision_score(y_test_antagonism, data13["antagonism"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_antagonism, data13["antagonism"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_antagonism, data13["antagonism"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data13["antagonism"])
        elif model_choisi == "MoritzLaurer/deberta-v3-large-zeroshot-v2.0":
             if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data21["antagonism"], labels =["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                metrics["Précision"] = precision_score(y_test_antagonism, data21["antagonism"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_antagonism, data21["antagonism"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_antagonism, data21["antagonism"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data21["antagonism"])
             elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data22["antagonism"], labels = ["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                 metrics["Précision"] = precision_score(y_test_antagonism, data22["antagonism"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test_antagonism, data22["antagonism"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test_antagonism, data22["antagonism"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data22["antagonism"])
             elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data23["antagonism"], labels = ["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                metrics["Précision"] = precision_score(y_test_antagonism, data23["antagonism"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_antagonism, data23["antagonism"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_antagonism, data23["antagonism"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data23["antagonism"])
        elif model_choisi == "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data31["antagonism"], labels =["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                metrics["Précision"] = precision_score(y_test_antagonism, data31["antagonism"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_antagonism, data31["antagonism"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_antagonism, data31["antagonism"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data31["antagonism"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data32["antagonism"], labels = ["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                 metrics["Précision"] = precision_score(y_test_antagonism, data32["antagonism"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test_antagonism, data32["antagonism"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test_antagonism, data32["antagonism"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data32["antagonism"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data33["antagonism"], labels = ["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                metrics["Précision"] = precision_score(y_test_antagonism, data33["antagonism"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_antagonism, data33["antagonism"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_antagonism, data33["antagonism"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data33["antagonism"])
        elif model_choisi == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data41["antagonism"], labels = ["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                metrics["Précision"] = precision_score(y_test_antagonism, data41["antagonism"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_antagonism, data41["antagonism"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_antagonism, data41["antagonism"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data41["antagonism"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data42["antagonism"], labels = ["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                 metrics["Précision"] = precision_score(y_test_antagonism, data42["antagonism"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test_antagonism, data42["antagonism"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test_antagonism, data42["antagonism"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data42["antagonism"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data43["antagonism"], labels = ["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                metrics["Précision"] = precision_score(y_test_antagonism, data43["antagonism"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_antagonism, data43["antagonism"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_antagonism, data43["antagonism"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data43["antagonism"])
        elif model_choisi == "MoritzLaurer/roberta-large-zeroshot-v2.0":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data51["antagonism"], labels = ["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                metrics["Précision"] = precision_score(y_test_antagonism, data51["antagonism"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_antagonism, data51["antagonism"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_antagonism, data51["antagonism"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data51["antagonism"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data52["antagonism"], labels = ["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                 metrics["Précision"] = precision_score(y_test_antagonism, data52["antagonism"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test_antagonism, data52["antagonism"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test_antagonism, data52["antagonism"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data52["antagonism"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data53["antagonism"], labels =["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                metrics["Précision"] = precision_score(y_test_antagonism, data53["antagonism"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_antagonism, data53["antagonism"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_antagonism, data53["antagonism"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data53["antagonism"])
        elif model_choisi == "MoritzLaurer/bge-m3-zeroshot-v2.0":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data61["antagonism"], labels = ["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                metrics["Précision"] = precision_score(y_test_antagonism, data61["antagonism"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_antagonism, data61["antagonism"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_antagonism, data61["antagonism"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data61["antagonism"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data62["antagonism"], labels = ["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                 metrics["Précision"] = precision_score(y_test_antagonism, data62["antagonism"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test_antagonism, data62["antagonism"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test_antagonism, data62["antagonism"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data62["antagonism"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test_antagonism, data63["antagonism"], labels =["Conciliant", "Résistant", "Opposant","Irréconciliant"])
                metrics["Précision"] = precision_score(y_test_antagonism, data63["antagonism"], average='macro')
                metrics["Rappel"] =   recall_score(y_test_antagonism, data63["antagonism"], average='macro')  
                metrics["Score F1"] = f1_score(y_test_antagonism, data63["antagonism"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test_antagonism, data63["antagonism"])
        return metrics
def evaluation_3 (model_choisi, prompt ):
        metrics = {"Matrice_confusion": "", "Précision": "", "Rappel": "", "Score F1" :"", "Accuracy": ""}
        if model_choisi == "facebook/bart-large-mnli":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data11["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data11["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data11["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data11["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data11["class"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test, data12["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                 metrics["Précision"] = precision_score(y_test, data12["class"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test, data12["class"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test, data12["class"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test, data12["class"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data13["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data13["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data13["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data13["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data13["class"])
        elif model_choisi == "MoritzLaurer/deberta-v3-large-zeroshot-v2.0":
             if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data21["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data21["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data21["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data21["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data21["class"])
             elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test, data22["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                 metrics["Précision"] = precision_score(y_test, data22["class"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test, data22["class"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test, data22["class"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test, data22["class"])
             elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data23["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data23["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data23["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data23["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data23["class"])
        elif model_choisi == "MoritzLaurer/deberta-v3-large-zeroshot-v2.0-c":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data31["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data31["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data31["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data31["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data31["class"])
            elif prompt =="prompt_2":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data32["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data32["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data32["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data32["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data32["class"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data33["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data33["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data33["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data33["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data33["class"])
        elif model_choisi == "MoritzLaurer/deberta-v3-base-zeroshot-v2.0":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data41["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data41["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data41["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data41["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data41["class"])
            elif prompt =="prompt_2":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data42["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data42["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data42["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data42["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data42["class"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data43["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data43["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data43["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data43["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data43["class"])
        elif model_choisi == "MoritzLaurer/roberta-large-zeroshot-v2.0":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data51["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data51["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data51["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data51["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data51["class"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test, data52["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                 metrics["Précision"] = precision_score(y_test, data52["class"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test, data52["class"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test, data52["class"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test, data52["class"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data53["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data53["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data53["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data53["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data53["class"])
        elif model_choisi == "MoritzLaurer/bge-m3-zeroshot-v2.0":
            if prompt == "prompt_1":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data61["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data61["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data61["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data61["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data61["class"])
            elif prompt =="prompt_2":
                 metrics["Matrice_confusion"] = confusion_matrix(y_test, data62["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                 metrics["Précision"] = precision_score(y_test, data62["class"], average='macro')
                 metrics["Rappel"] =   recall_score(y_test, data62["class"], average='macro')  
                 metrics["Score F1"] = f1_score(y_test, data62["class"], average='macro')
                 metrics ["Accuracy" ] = accuracy_score(y_test, data62["class"])
            elif prompt == "prompt_3":
                metrics["Matrice_confusion"] = confusion_matrix(y_test, data63["class"], labels =["Passif", "Aligné", "Engagé","Moteur","Déchiré", "Opposant", "Irréductible"])
                metrics["Précision"] = precision_score(y_test, data63["class"], average='macro')
                metrics["Rappel"] =   recall_score(y_test, data63["class"], average='macro')  
                metrics["Score F1"] = f1_score(y_test, data63["class"], average='macro')
                metrics ["Accuracy" ] = accuracy_score(y_test, data63["class"])
        return metrics

def evolution_f1_score(model):

    fig = plt.figure(figsize=(10, 6))

  

    x = ["Prompt_1", "Prmopt_2", "Prompt_3"]
    y = [evaluation_3(model, "prompt_1")["Score F1"] , evaluation_3(model, "prompt_2")["Score F1"], evaluation_3(model, "prompt_3")["Score F1"] ]
    plt.plot(x, y, label=model, color = "green",marker='o', linestyle='--')

# Ajouter les légendes et les titres
    plt.xlabel('Prompts')
    plt.ylabel('F1-score macro')
    plt.title(f'Evolution du F1-score macro  ')
    plt.legend()
    return fig




             
             
                      





             
             
                      

        
   
