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

=> 7h comme pic de fraude est une vraie insight



une partition c'est une partie de topic, 2. chaque cunsummer va lire une partition , 3. on a fait la conversion de type car on a kafka n'accepte et ne comprit que les bytes , 4. consummer lag est la difference entre le dernier message produit et le dernier consommée , c'est pour ça il est considérer un métrique critique car c'est un indice fort sur la preformance du production 


Le DAG est une représentation de ces dépendances et de l'ordre dans lequel les transformations doivent être réalisées.
Le lineage, c'est le chemin parcouru par la donnée.

La data leakage (fuite de données) en Machine Learning, c’est quand des informations que le modèle ne devrait pas connaître au moment de la prédiction se retrouvent dans ses données d’entraînement.

BigQuery = Offline Store : Il contient l'historique.

Redis = Online Store : Redis sert à récupérer rapidement les features nécessaires au serving.

Feast est la couche qui définit et gère les features et leur accès entre ces deux mondes.

imbalanced-learn (souvent importé sous le nom imblearn) est une bibliothèque Python spécialement conçue pour gérer les jeux de données déséquilibrés en Machine Learning.

Elle complète scikit-learn en fournissant des algorithmes de rééchantillonnage (resampling) pour rééquilibrer la répartition entre la classe minoritaire et la classe majoritaire.

Le Rappel (Recall) répond à la question : "Sur 100 vraies fraudes commises, combien mon système a-t-il réussi à intercepter ?"


1. Precision (Précision) : 68.97% (0.6897)Ce que cela signifie : Quand le modèle prédit qu'une transaction est frauduleuse, il a raison dans 68,97 % des cas.Impact métier : Une précision élevée évite de bloquer la carte bancaire de vrais clients par erreur (fausses alertes/faux positifs).

2. Recall (Rappel / Sensibilité) : 2.90% (0.0290)Ce que cela signifie : Sur 100 vraies fraudes qui ont eu lieu, le modèle n'a réussi à en détecter que 2,9 % (il en a raté 97,1 %).Impact métier : C'est le point faible actuel de votre modèle. Le rappel très bas signifie que la majorité des fraudes passent entre les mailles du filet.

3. F1-Score : ~0.0557 (Calculé)Ce que cela signifie : C'est la moyenne harmonique entre la Precision et le Recall :$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.6897 \times 0.0290}{0.6897 + 0.0290} \approx 0.0557$$Impact métier : Comme le rappel est très bas, le F1-Score s'effondre. Il indique que l'équilibre global entre "ne pas se tromper" et "attraper les fraudeurs" n'est pas bon.

4. AUC-ROC : 0.8185Ce que cela signifie : Mesure la capacité du modèle à classer une transaction frauduleuse avec un score de risque plus élevé qu'une transaction légitime.Impact métier : Un score de 81,85 % indique que le modèle a un bon pouvoir de discrimination global. Le modèle sépare bien les probabilités, mais le seuil de décision par défaut (0.5) est actuellement trop strict, ce qui explique le Recall très bas.

5. AUC-PR (Area Under Precision-Recall Curve) : 0.2375Ce que cela signifie : Mesure la qualité des prédictions en se concentrant uniquement sur la classe minoritaire (les fraudes).Impact métier : C'est la métrique de référence pour la fraude. Un score de 0.2375 est bien supérieur au niveau aléatoire (qui serait égal au taux de fraude de 0.035 soit 3.5 %), mais montre qu'il reste de la marge d'amélioration.