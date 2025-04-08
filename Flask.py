from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

app = Flask(__name__)

# Charger le modèle et le scaler
model = load_model(r'C:\Users\Rouamba Abdoul-Hakim\Documents\GitHub\IA-applying-to-Industry-4.0\best_modelLSTM.h5')
scaler = joblib.load(r'C:\Users\Rouamba Abdoul-Hakim\Documents\GitHub\IA-applying-to-Industry-4.0\data_scaler (1).pkl')

# Initialiser le LabelEncoder avec les mêmes paramètres qu'à l'entraînement
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['no', 'yes'])  # Ajustez selon vos données réelles

# Définir les colonnes utilisées dans l'entraînement
FEATURES = ['hour', 'minute', 'second', 'grip_lost'] + \
           [col for col in pd.read_excel('dataset_02052023.xlsx').columns 
            if col not in ['Timestamp', 'Robot_ProtectiveStop'] and 
            pd.read_excel('dataset_02052023.xlsx')[col].dtype in ['float64', 'int64']]

NUMERIC_COLS = [col for col in FEATURES if col not in ['grip_lost', 'hour', 'minute', 'second']]

# Charger les moyennes des colonnes numériques (à créer pendant l'entraînement)
try:
    mean_values = joblib.load('mean_values.pkl')  # Fichier à créer pendant l'entraînement
except FileNotFoundError:
    # Si le fichier n'existe pas, définir des valeurs par défaut (à ajuster selon vos données)
    mean_values = pd.Series(0, index=NUMERIC_COLS)  # Remplacez par les vraies moyennes si possible

def preprocess_data(data, time_steps=10):
    """
    Prétraitement des données suivant exactement les étapes de l'entraînement
    """
    try:
        # Convertir en DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        # Vérifier les colonnes
        missing_cols = set(FEATURES) - set(df.columns)
        if missing_cols:
            return None, f"Colonnes manquantes : {missing_cols}"

        # Étape 1: Nettoyage et conversion du timestamp (si présent)
        if 'Timestamp' in df.columns:
            df['Timestamp'] = df['Timestamp'].astype(str).str.strip('"')
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df['hour'] = df['Timestamp'].dt.hour
            df['minute'] = df['Timestamp'].dt.minute
            df['second'] = df['Timestamp'].dt.second
            df = df.drop(columns=['Timestamp'])

        # Étape 2: Encodage de grip_lost
        if df['grip_lost'].dtype in ['object', 'str']:
            df['grip_lost'] = label_encoder.transform(df['grip_lost'])
        elif df['grip_lost'].max() > 1 or df['grip_lost'].min() < 0:
            df['grip_lost'] = label_encoder.transform(df['grip_lost'].map({0: 'no', 1: 'yes'}))

        # Étape 3: Gestion des valeurs manquantes
        # Utiliser les moyennes du scaler d'entraînement
        df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(pd.Series(mean_values, index=scaler.feature_names_in_))

        # Étape 4: Normalisation
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

        # Étape 5: Création des séquences
        if len(df) < time_steps:
            return None, f"Pas assez de données pour créer une séquence de {time_steps} pas"

        X = df[FEATURES].values
        X_seq = []
        for i in range(len(X) - time_steps + 1):
            X_seq.append(X[i:i + time_steps])
        
        return np.array(X_seq), None

    except Exception as e:
        return None, f"Erreur de préprocessing : {str(e)}"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'Aucune donnée fournie'}), 400

        # Prétraiter les données
        X_seq, error = preprocess_data(data)
        
        if error:
            return jsonify({'error': error}), 400

        # Faire la prédiction
        predictions_proba = model.predict(X_seq)
        predictions = (predictions_proba > 0.5).astype(int).flatten()

        # Préparer la réponse
        response = {
            'predictions': predictions.tolist(),
            'probabilities': predictions_proba.flatten().tolist()
        }
        
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)