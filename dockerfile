# Utiliser une image de base Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier le fichier requirements.txt (à créer ensuite)
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tous les fichiers nécessaires dans le conteneur
COPY . .

# Exposer le port sur lequel l'API va tourner
EXPOSE 5000

# Commande pour lancer l'application
CMD ["python", "Flask.py"]