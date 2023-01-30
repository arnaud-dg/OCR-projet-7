# -----------------------------------------------------------------------------------------
# -----------------------     Création de l'API avec FLASK   ------------------------------
# -----------------------------------------------------------------------------------------

# API servant à charger les données des clients, à prédire le score à travers le modèle et à produire les SHAP values

# Import des librairies requises
import flask
from flask import jsonify
import pickle
import numpy as np
import pandas as pd
import shap
import os

# Initialisation de l'API
app = flask.Flask(__name__)
app.config["DEBUG"] = True

# Chemin du repertoire où sont stockés les fichiers
path = os.path.dirname(os.path.realpath(__file__))

# Page accueil
@app.route('/', methods=['GET'])
def home():
    return """
<h1>Projet #7 - Implémentez un modèle de scoring (A.Duigou)</h1>
<p>Le but de cet API est de:
 - charger les données clients
 - charger le modèle de prédiction
 - prédire le scoring d'un client
 - produire les shap values pour expliquer la prédiction
 </p>
"""

# Chargement du modèle optimisé enregistré sous forme binaire/pickle
model = pickle.load(open(path + "/model_2.pkl","rb"))
# Chargement des données test (chargement d'un échantillon uniquement)
X = pd.read_csv(path + "/data_sample.csv")

# Création d'une route pour récupérer les données clients
@app.route("/data", methods=["GET"])
def get_data():
    df_all = X.to_dict("list")
    return jsonify(df_all)

# Création d'une route pour récupérer les données clients et effectuer les prédictions
@app.route("/data/client/<client_id>", methods=["GET"])
def client_data(client_id):
    # Filtre des données sur la base de l'ID client
    df_sample = X[X["SK_ID_CURR"] == int(client_id)]
    feature_array = np.asarray(df_sample.iloc[0,1:21])
    # Calculer la prédiction et la probabilité pour le client
    df_sample["prediction"] = model.predict([feature_array]).tolist()[0]
    df_sample['proba_1'] = model.predict_proba([feature_array])[:,1].tolist()[0]
    # Calcul de la feature importance
    explainer = shap.KernelExplainer(model.predict_proba, X.iloc[:,1:21]) 
    shap_values = explainer.shap_values(feature_array, l1_reg="aic")
    # Ajout des SHAP values dans le Dataframe
    df_sample["expected"] = explainer.expected_value[1]
    new_line = [99999] + list(shap_values[1]) + [0,0,explainer.expected_value[1]]
    df_sample.loc[1] = new_line
    # Creation d'un dictionnaire 
    sample = df_sample.to_dict("list")
    # Retourne la prédiction sous forme d'un fichier json
    return jsonify(sample)

# A retirer en ligne :
if __name__ == '__main__':
    app.run()
#     app.run(host='0.0.0.0')