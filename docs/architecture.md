Le problème que ce projet résout : 

  Les banques traitent des millions de transactions chaque jour, et les décisions de détection de la fraude doivent être prises en quelques millisecondes. Les données brutes des transactions ne peuvent pas être transmises directement à un modèle d'apprentissage automatique, car elles manquent souvent de caractéristiques pertinentes. Mon projet consiste à développer une plateforme de caractéristiques en temps réel qui ingère les transactions en flux continu, génère des caractéristiques cohérentes, les stocke dans un « feature store » et les fournit à un modèle de détection de la fraude. Cette approche garantit la rapidité des prédictions, la cohérence du calcul des caractéristiques entre les phases d'entraînement et de production, ainsi qu'une architecture capable de monter en charge pour traiter d'importants volumes de transactions.

Les 3 couches de l'architecture :

les couches de l'architecture complet de projet: 

Couche 1 : Ingestion Layer (Kafka)

La transaction arrive.
Couche 2 : Feature Engineering

Maintenant on ouvre le colis.
Couche 3 : Feature Store (Feast)

Maintenant que les features sont créées, où les stocker ? Dans Feast.
Couche 4 : Prediction (Serving)

Le modèle reçoit maintenant les données
Couche 5 : API (FastAPI)

Le modèle a pris une décision. FastAPI répond à la banque.

                    👤 Client
                       │
                       ▼
            Paiement par carte bancaire
                       │
                       ▼
             ┌──────────────────┐
             │ 1. Kafka          │
             │ Reçoit le flux    │
             └──────────────────┘
                       │
                       ▼
             ┌──────────────────┐
             │ 2. Feature        │
             │ Engineering       │
             │ (Hour, Log, ...)  │
             └──────────────────┘
                       │
                       ▼
             ┌──────────────────┐
             │ 3. Feast          │
             │ Feature Store     │
             └──────────────────┘
                       │
                       ▼
             ┌──────────────────┐
             │ 4. Modèle ML      │
             │ XGBoost           │
             └──────────────────┘
                       │
                       ▼
             ┌──────────────────┐
             │ 5. FastAPI        │
             │ Réponse à la      │
             │ banque            │
             └──────────────────┘
                       │
                       ▼
         Paiement accepté ou bloqué

Pourquoi on a besoin d'un feature store: 
on est besoin d'utiliser un feature store pour stocker les features créees pour on garde la méme façon de l'entrainement et la production , comme une bibliothque ou on stocke les features pour le réutiliser sans le recalculer 