import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# URL de base de l'API
BASE_URL = "http://localhost:5000"

def generate_test_data(n_samples=15):
    """
    Génère des données de test similaires à votre dataset
    """
    start_time = datetime(2023, 5, 2, 14, 30, 0)
    timestamps = [start_time + timedelta(seconds=i) for i in range(n_samples)]
    
    data = {
        'Timestamp': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
        'grip_lost': np.random.choice(['no', 'yes'], n_samples).tolist(),
        'cycle ': np.random.randint(1, 100, n_samples).tolist(),
        'Num': np.random.randint(1, 1000, n_samples).tolist(),
        'Temperature_J0': np.random.normal(25, 5, n_samples).tolist(),
        'Temperature_J1': np.random.normal(25, 5, n_samples).tolist(),
        'Temperature_J2': np.random.normal(25, 5, n_samples).tolist(),
        'Temperature_J3': np.random.normal(25, 5, n_samples).tolist(),
        'Temperature_J4': np.random.normal(25, 5, n_samples).tolist(),
        'Temperature_J5': np.random.normal(25, 5, n_samples).tolist(),
        'Temperature_T0': np.random.normal(25, 5, n_samples).tolist(),
        'Current_J0': np.random.uniform(0, 10, n_samples).tolist(),
        'Current_J1': np.random.uniform(0, 10, n_samples).tolist(),
        'Current_J2': np.random.uniform(0, 10, n_samples).tolist(),
        'Current_J3': np.random.uniform(0, 10, n_samples).tolist(),
        'Current_J4': np.random.uniform(0, 10, n_samples).tolist(),
        'Current_J5': np.random.uniform(0, 10, n_samples).tolist(),
        'Speed_J0': np.random.uniform(0, 100, n_samples).tolist(),
        'Speed_J1': np.random.uniform(0, 100, n_samples).tolist(),
        'Speed_J2': np.random.uniform(0, 100, n_samples).tolist(),
        'Speed_J3': np.random.uniform(0, 100, n_samples).tolist(),
        'Speed_J4': np.random.uniform(0, 100, n_samples).tolist(),
        'Speed_J5': np.random.uniform(0, 100, n_samples).tolist(),
        'Tool_current': np.random.uniform(0, 5, n_samples).tolist(),
        'hour': np.random.uniform(0, 100, n_samples).tolist(),
        'minute': np.random.uniform(0, 100, n_samples).tolist(),
        'second': np.random.uniform(0, 100, n_samples).tolist(),
    }
    
    return pd.DataFrame(data).to_dict(orient='records')

def test_health_endpoint():
    """Teste le endpoint /health"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("Test Health Endpoint:")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}\n")
        return response.status_code == 200
    except Exception as e:
        print(f"Erreur lors du test health: {str(e)}\n")
        return False

def test_predict_endpoint(data):
    """Teste le endpoint /predict avec les données fournies"""
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            f"{BASE_URL}/predict",
            data=json.dumps(data),
            headers=headers
        )
        
        print("Test Predict Endpoint:")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Predictions: {result['predictions']}")
            print(f"Probabilities: {result['probabilities']}\n")
        else:
            print(f"Erreur: {response.json()}\n")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Erreur lors du test predict: {str(e)}\n")
        return False



def main():
    # Vérifier si l'API est en marche
    print("=== Début des tests ===\n")
    
    if not test_health_endpoint():
        print("L'API ne semble pas fonctionner. Veuillez la démarrer d'abord.")
        return
    
    # Générer et tester avec des données valides
    test_data = generate_test_data()
    print("Données de test générées:")
    print(json.dumps(test_data[:2], indent=2))  # Afficher les 2 premières lignes
    print(f"Nombre total d'échantillons: {len(test_data)}\n")
    
    test_predict_endpoint(test_data)
    
    print("=== Fin des tests ===")

if __name__ == "__main__":
    # Assurez-vous que l'API Flask est en cours d'exécution avant de lancer ce script
    main()