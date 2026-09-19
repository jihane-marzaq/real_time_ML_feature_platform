FROM python:3.13-slim

WORKDIR /app

# Copie requirements en premier — layer caching Docker
# Si le code change mais pas les deps, cette layer reste en cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie le code
COPY app/ ./app/

# Variables d'environnement par défaut
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Port exposé
EXPOSE 8000

# Commande de démarrage
# --workers 1 : un seul worker car le modèle est en mémoire
# En production multi-instance, chaque container a son propre modèle
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]