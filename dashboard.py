# -----------------------------------------------------------------------------------------
# -----------------------    Création du dashboard Streamlit ------------------------------
# -----------------------------------------------------------------------------------------

# Import des librairies
# Génériques
import json
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
# Streamlit
import streamlit as st
import streamlit.components.v1 as components
from urllib.request import urlopen
# Sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
# SHAP values
import shap

# Chargment des images intégrées à la page
logo_page =  Image.open("./Images/euro.png")
logo_entreprise = Image.open("./Images/pret_a_depenser.png")
logo_credit = Image.open("./Images/credit.png")

# Paramétrage de la page sur streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config( 
    page_title="Pret a dépenser - Evaluation du risque de crédit",
    page_icon=logo_page,
    layout="wide" # Affichage élargi sur l'entiereté de la page
)
# Configuration de l'API
# en local : API_url = "http://192.168.1.94:5000/"
# en ligne :
API_url = "https://api-flask-ocr-projet-7.herokuapp.com/"

# Initialisation de javascript pour l'affichage des SHAP values
shap.initjs()
# Fonctions fournissant les SHAP values
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Titre du Dashboard
st.title("Pret a dépenser - Evaluation du risque de crédit") 
st.subheader("Le rapport graphique qui vous permet d'expliquer au client notre décision vis-à-vis de sa demande de prêt. :moneybag:")

# -----------------------------------------------------------------------------------------
# ---------------      Récupération des données (dataset complet)     ---------------------
# -----------------------------------------------------------------------------------------

# Récupération des données à travers l'API
json_url_all=urlopen(API_url + "data") 
API_data_all=json.loads(json_url_all.read()) # Chargement de données JSON de l'API
data_all=pd.DataFrame(API_data_all) # Conversion en dataframe

# Création de la liste des clients
client_list=data_all["SK_ID_CURR"].tolist()
# Création de la liste des colonnes
columns=list(data_all.drop(columns="SK_ID_CURR").columns)
# Préparation des données pour les graphiques de comparaison
data_plot=data_all[columns]

# Création de la liste des booléens pour l'affichage graphique
categories=[]
for col in columns:
    if len(data_plot[col].value_counts().index) == 2:
        if (np.sort(data_plot[col].value_counts().index).astype(int) == [0, 1]).all():
            categories.append(col)
# Création de la liste pour les colonnes catégorielles
col_one=[]
col_std=[]
for col in data_plot.columns:
    if "_cat_" in col or col in categories:
        col_one.append(col)
    else:
        col_std.append(col)
# Reordonnancement des colonnes
columns=col_std + col_one

# Mise en place du standardscaler pour transformer les données à travers le pipeline
# Application du scaler
data_plot_std=scale_min_max.fit_transform(data_plot)
# génération du dataframe
data_plot_final=pd.DataFrame(data_plot_std, columns=columns)

# ----------------- Données de la population  pour comparaison ------------------------
# Création des données de référence (moyenne, médiane, mode)
Z=data_all[columns]
data_population=pd.DataFrame(index=Z.columns)
data_population["mean"] = Z.mean()
data_population["median"] = Z.median()
data_population["mode"] = Z.mode().iloc[0, :]
data_population = data_population.transpose()
# Retrait des valeurs non pertinentes
for col in data_population.columns:
    if col in col_one:
        data_population.loc["median", col] = np.NaN
    else:
        data_population.loc["mode", col] = np.NaN

# -----------------------------------------------------------------------------------------
# -----------------------       Bandeau latéral gauche       ------------------------------
# -----------------------------------------------------------------------------------------

st.sidebar.image(logo_entreprise, width=250)
st.sidebar.title("Informations sur le demandeur") 
# Saisie de l'ID client
client_id=st.sidebar.selectbox("Saisissez l'identifiant client pour afficher le rapport:", client_list)

# Recherche de l'index du client dans le dataset
client_idx=data_all[data_all["SK_ID_CURR"] == client_id].index
# Filtre du dataset sur le client ID
data_client=data_plot_final.loc[client_idx, :]

# Affichage des colonnes concernées
aff_par_defaut=['EXT_SOURCE_2',
         'EXT_SOURCE_3',
         'BURO_DAYS_CREDIT_MIN',
         'BURO_DAYS_CREDIT_ENDDATE_MIN',
         'NAME_INCOME_TYPE_cat_Working',
         'BURO_CREDIT_ACTIVE_cat_Active_MEAN',
         'DAYS_BIRTH',
         'DAYS_EMPLOYED',
         'CODE_GENDER',
         'DAYS_LAST_PHONE_CHANGE']
columns_selected = st.sidebar.multiselect("Informations du client à afficher",columns, aff_par_defaut)

# Creation de listes pour les différentes colonnes saisies
columns_categ = []
columns_quanti = []
for col in columns:
    if col in columns_selected:
        if col in categories:
            columns_categ.append(col)
        else:
            columns_quanti.append(col)

# Paramétrage de l'affichage des SHAP values
option_SHAP=st.selectbox('Quel graphique d\'interprétabilité désirez-vous?',('Forme simple', 'En cascade', 'Linéaire'))

# -----------------------------------------------------------------------------------------
# -----------------------          Page principale           ------------------------------
# -----------------------------------------------------------------------------------------

# Commentaires et introduction
col1, col2 = st.columns([1, 2])
with col1: 
    st.image(logo_credit, width=400) 
with col2: 
    st.write("""Cet interface utilisateur vous guide afin d'expliquer au client notre positionnement par rapport 
    à sa demande de prêt. Nos équipes d'analystes ont analysé la demande du client dans le respect des droits bancaires et du réglèment RGPD.""")
    st.write("""Saisissez l'identifiant client dans le bandeau latéral gauche, ainsi que les dcritèress que vous souhaitez voir afficher""")
    st.write("""Le rapport détaillé se décompose en 3 parties :""")
    st.write("""1. Le score client et le niveau de risque associé""")
    st.write("""2. Son positionnement par rapport à la population des clients de notre base de donnée""")
    st.write("""3. Les crtières qui dans son cas de figure conduisent à notre position""")

# Affichage du rapport
with st.spinner("Traitement en cours..."):

# -----------------------------------------------------------------------------------------
# --------------      Récupération des données (client spécifique)     --------------------
# -----------------------------------------------------------------------------------------

    # Récupération des données clients depuis l'API
    json_url_client=urlopen(API_url + "data/client/" + str(client_id))
    API_data_client=json.loads(json_url_client.read())
    df=pd.DataFrame(API_data_client)

    # Liste des colonnes inutiles
    columns_info=["SK_ID_CURR", "expected", "prediction", "proba_1"]
    
    # Stockage des colonnes utilisées pour les SHAP values
    client_data=df.drop(columns = columns_info).iloc[0:1,:]
    features_analysis=client_data.columns
    
    # Stockage des données requises pour les SHAP values
    data_explain=np.asarray(client_data)
    shap_values=df.drop(columns = columns_info).iloc[1,:].values
    expected_value=df["expected"][0]
    
    # Affichage du score client
    st.subheader("Scoring client (en %) :")    
    # Affichage avec échelle visuelle du score client
    st.slider("", min_value=0, max_value=100, value=int(round(df["proba_1"][0],2)*100), disabled=True)
    # Affichage avec échelle visuelle du score client
    col1, col2, col3 = st.columns(3)
    if df["proba_1"][0]<0.45:
        with col1:
            st.success("Niveau de risque faible")
    elif df["proba_1"][0]>0.55:
        with col3:
            st.error("Niveau de risque élevé")
    else:
        with col2:
            st.warning("Niveau de risque modéré")

# -----------------------------------------------------------------------------------------
# -----------------------      Interprétation SHAP values     -----------------------------
# -----------------------------------------------------------------------------------------

    
    st.subheader("Interprétabilité des résultats - Rapport détaillé")
    
    if option_SHAP=='Forme simple':
        # Affichage du SHAP force_plot pour le client
        fig_force = shap.force_plot(expected_value, shap_values, data_explain, feature_names=features_analysis) 
        st_shap(fig_force)
    elif option_SHAP=='En cascade':
        # Affichage du SHAP waterfall pour le client
        fig_water = shap.plots._waterfall.waterfall_legacy(expected_value, shap_values, feature_names=features_analysis, max_display=10)
        st.pyplot(fig_water)
    else:
        # Decision plot
        fig_decision = shap.decision_plot(expected_value, shap_values, features_analysis)
        st.pyplot(fig_decision)


# -----------------------------------------------------------------------------------------
# -----------------------      Informations sur le client     -----------------------------
# -----------------------------------------------------------------------------------------
        
    st.subheader("Caractéristiques du client :")
    
    # Display plots that compare the current client within all the clients
    # For quantitative features first
    # Initialize the figure
    f, ax = plt.subplots(figsize=(7, 5))
    # Set the style for average values markers
    meanpointprops = dict(markeredgecolor="black", markersize=8, markerfacecolor="green", markeredgewidth=0.66)
    # Affichage d'un boxplot pour chaque donnée retenue
    sns.boxplot(
            data=data_plot_final[columns_quanti],
            orient="h",
            whis=3,
            palette="pastel",
            linewidth=0.7,
            width=0.6,
            showfliers=False,
            showmeans=True,
            meanprops=meanpointprops)
    # Add in a point to show current client
    sns.stripplot(
            data=data_client[columns_quanti],
            orient="h",
            size=4,
            # palette="blend:firebrick,firebrick",
            palette=["seagreen", "black"],
            marker="D",
            edgecolor="black",
            linewidth=0.66)
    # RRtrait de l'axe x
    ax.set_xticklabels([])
    # Retrait des lignes des axes
    sns.despine(trim=True, left=True, bottom=True, top=True)
    # Retrait des gradations
    plt.tick_params(left=False, bottom=False)
    # Mise en forme de l'axe y
    ax.set_yticklabels(columns_quanti,
            fontdict={"fontsize": "medium",
                "fontstyle": "italic",
                "verticalalignment": "center",
                "horizontalalignment": "right"})
    # Séparateurs pour les lignes y
    lines = [ax.axhline(y, color="grey", linestyle="solid", linewidth=0.7)
                            for y in np.arange(0.5, len(columns_quanti)-1, 1)]

    # Proxy artists to add a legend
    average = mlines.Line2D([], [], color="black", marker="^", linestyle="None", markeredgecolor="black",
                            markeredgewidth=0.66, markersize=8, label="moyenne")
    current = mlines.Line2D([], [], color="seagreen", marker="D", linestyle="None", markeredgecolor="black",
                            markeredgewidth=0.66, markersize=8, label="client courant")
    ax.legend(handles=[average, current], bbox_to_anchor=(1, 1), fontsize="small")
    plt.title("Positionnement du client dans la population")
    # Affichage du graphique
    st.pyplot(f)
    
    # Then for categories
    # First ceate a summary dataframe
    df_plot_cat = pd.DataFrame()
    for col in columns_categ:
        df_plot_cat = pd.concat(
            [
                df_plot_cat,
                pd.DataFrame(data_plot_final[col].value_counts()).transpose(),
            ]
        )
    df_plot_cat["categories"] = df_plot_cat.index
    df_plot_cat = df_plot_cat[["categories", 0.0, 1.0]]
    df_plot_cat = df_plot_cat.fillna(0)
    # Then create the plot
    with plt.style.context("_mpl-gallery-nogrid"):
        # plot a Stacked Bar Chart using matplotlib
        ax = df_plot_cat.plot(
            x="categories",
            kind="barh",
            stacked=True,
            mark_right=True,
            grid=False,
            xlabel="",
            figsize=(6, 0.5 * len(columns_categ)),
        )
        # Display percentages of each value
        df_total = df_plot_cat[0.0] + df_plot_cat[1.0]
        df_rel = df_plot_cat[df_plot_cat.columns[1:]].div(df_total, 0) * 100
        for n in df_rel:
            for i, (cs, ab, pc) in enumerate(
                zip(df_plot_cat.iloc[:, 1:].cumsum(1)[n], df_plot_cat[n], df_rel[n])
            ):
                plt.text(
                    cs - ab / 2,
                    i,
                    str(np.round(pc, 1)) + "%",
                    va="center",
                    ha="center",
                    color="white",
                )
        # Display markers for the current client
        comparison = []
        for col in columns_categ:
            total = len(data_plot_final[col])
            client_val = int(data_client[col])
            mask = data_plot_final[col] == client_val
            temp = data_plot_final[mask]
            count = temp[col].value_counts().values[0]
            comparison.append(client_val * (total - count) + count / 2 + 15)
        plt.plot(
            comparison,
            columns_categ,
            marker="D",
            color="firebrick",
            markersize=8,
            markeredgecolor="black",
            linestyle="None",
            markeredgewidth=0.66,
        )
        # Manage display
        sns.despine(
            trim=True,
            left=True,
            bottom=False,
            top=True,
        )
        plt.legend(
            ncols=1,
            labels=["client courant", 0, 1],
            bbox_to_anchor=(1, 1),
            fontsize="small",
        )
        ax.set_yticklabels(
            columns_categ,
            fontdict={
                "fontsize": "medium",
                "fontstyle": "italic",
                "verticalalignment": "center",
                "horizontalalignment": "right",
            },
        )
        plt.xlabel("Population")
        plt.title("Informations catégorielles")
    st.pyplot()
    
    # in an expander, display the client's data and comparison with average
    with st.expander("Ouvrir pour afficher les données détaillées"):
        temp_df = pd.concat([client_data, data_population])
        new_df = temp_df.transpose()
        new_df.columns = ["Client (" + str(client_id) + ")", "Moyenne",
                          "Médiane", "Mode"]
        st.table(new_df.loc[columns_selected,:])

# Display a success message in the sidebar once the process is completed
with st.sidebar:
    end = datetime.datetime.now()
    text_success = "Last successful run : " + str(end.strftime("%Y-%m-%d %H:%M:%S"))
    st.success(text_success)
