# IA-applying-to-Industry-4.0
Prédiction des Arrêts de Protection d’un Cobot UR3
Réalisé par:
KANOHA ELENGA Helmie Naella Jihane
ROUAMBA Abdoul Hakim

Contexte
Dans le cadre du module Intelligence Artificielle et Applications en Industrie 4.0, nous avons réalisé un projet de prédiction des arrêts de protection d’un cobot UR3. L’objectif principal est d’anticiper les arrêts imprévus à partir des données issues des capteurs du cobot, afin d’améliorer la maintenance prédictive et éviter les interruptions de production.

Objectif

L’objectif est de prédire la variable cible « Protective Stops » à partir des données suivantes, extraites des 10 dernières unités de temps:

_ Courants électriques des articulations (J0_C à J5_C)

_ Températures des articulations (J0_T à J5_T)

_ Vitesses des articulations (J0_V à J5_V)

_ Courant de la pince (Gripper_C)

_ Nombre de cycles d’opération (Operation Cycles)

_ Pertes de préhension (Grip Losses)

Méthodologie
1. Exploration et Prétraitement
Chargement du dataset UR3 CobotOps

Nettoyage et normalisation des données

Construction des séquences temporelles (fenêtres de taille 10)

Séparation du jeu de données en ensemble d’entraînement, validation et test

2. Modélisation
Nous avons testé plusieurs modèles d’apprentissage machine et deep learning.
Le modèle principal utilisé est un LSTM (Long Short-Term Memory), bien adapté à la nature séquentielle des données.
Nous avons également comparé ses performances à d’autres modèles tels que :

Arbre de décision

Random Forest

3. Optimisation
Plusieurs tests ont été réalisés pour optimiser les hyperparamètres :

Taille des batchs

Nombre de couches LSTM

Fonctions d’activation

L’utilisation de Keras Tuner a permis de systématiquement tester différentes combinaisons et d’identifier la configuration offrant les meilleures performances sur l’ensemble de validation.

Métriques utilisées : accuracy, recall, précision, F1-score
Le suivi des expériences a été réalisé avec Weights & Biases (WandB).

4. Déploiement
Le modèle final est intégré dans une API Flask qui permet d’envoyer une séquence et d’obtenir une prédiction.
L’API est conteneurisée avec Docker pour faciliter son déploiement sur tout environnement.

Résultats
Le modèle LSTM a obtenu de très bons résultats, avec une précision 0,9596 sur le jeu de test.
Il a permis d’anticiper efficacement les arrêts de protection en se basant sur les tendances des capteurs.

Structure du projet

📦 projet-cobot/
├── data/                 # Dataset UR3
├── notebooks/            # Analyse, prétraitement, modélisation
├── model/                # Modèle entraîné
├── api/                  # API Flask
│   ├── app.py
│   └── model_loader.py
├── Dockerfile
├── requirements.txt
└── README.md
