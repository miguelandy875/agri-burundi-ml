# Rapport : Déploiement d'un Modèle de Prédiction Agricole au Burundi

## Analyse du Scénario 3 — Gitega, Haricot, Pluviométrie 430mm (Q29)

Le scénario 3 correspond à une parcelle de haricot située à Gitega, à 1 720 m d'altitude, avec une pluviométrie de 430 mm, une température moyenne de 18,2 °C, sans engrais et sans irrigation. Après transformation avec le même pipeline que celui utilisé à l'entraînement, les trois modèles prédisent une bonne récolte. L'arbre de décision donne une probabilité d'environ 97,9 %, la forêt aléatoire environ 86,0 %, et la régression logistique environ 90,0 %.

Cette prédiction ne signifie pas que le risque agronomique est faible. Une vérification des données nettoyées montre qu'il n'existe aucun exemple de haricot avec une pluviométrie inférieure ou égale à 430 mm. Dans la tranche 400-600 mm, on observe seulement 11 cas de haricot, dont 10 sont étiquetés comme bonnes récoltes. Pour Gitega-Haricot, les 18 observations disponibles sont toutes classées comme bonnes récoltes, avec des pluviométries à partir d'environ 640 mm. Le modèle extrapole donc hors de la zone réellement représentée dans les données d'entraînement.

Sur le plan agronomique, le haricot demande généralement entre 400 et 600 mm d'eau pendant son cycle. Une pluviométrie de 430 mm se situe près de la limite basse. Elle n'implique pas automatiquement une mauvaise récolte, mais elle indique un risque de stress hydrique, surtout si les pluies sont mal réparties. L'absence d'irrigation et d'engrais renforce ce risque. La prédiction favorable doit donc être comprise comme un résultat du dataset simulé, qui ne pénalise pas suffisamment les faibles pluviométries pour le haricot, plutôt que comme une vérité agronomique générale.

Trois recommandations pratiques peuvent être proposées. Premièrement, si l'eau est accessible, une irrigation complémentaire devrait être envisagée pendant la floraison et la formation des gousses. Deuxièmement, l'agriculteur pourrait utiliser des variétés de haricot à cycle court, mieux adaptées au stress hydrique. Troisièmement, des techniques comme le paillage, le travail minimum du sol et la couverture végétale peuvent limiter l'évaporation.

## Recommandation de Modèle pour Déploiement en Production (Q30)

La comparaison des modèles montre que la régression logistique obtient les meilleurs résultats globaux dans cette étude. D'après `metrics.json`, elle atteint une accuracy de 0,9399, un F1-score pondéré de 0,9203 et une AUC de 0,8358. Ces valeurs sont supérieures à celles de la forêt aléatoire pour le F1-score et l'AUC, et supérieures à celles de l'arbre de décision pour l'accuracy et l'AUC. En validation croisée à 5 folds, elle obtient environ 0,9373 ± 0,0085, ce qui indique une performance stable.

La forêt aléatoire obtient une accuracy de 0,9304, un F1-score de 0,8999 et une AUC de 0,7419. Sa validation croisée est très stable, avec environ 0,9341 ± 0,0040, mais son AUC plus faible indique une capacité de discrimination moins bonne. L'arbre de décision obtient une accuracy de 0,8987, un F1-score de 0,9009 et une AUC de 0,7520, avec une validation croisée d'environ 0,9254 ± 0,0141. Il est plus facile à expliquer, mais il est moins performant et plus sensible au choix de la profondeur.

Sur la base de ces résultats, je recommanderais de déployer la régression logistique comme modèle principal. Elle présente le meilleur compromis entre performance, interprétabilité et simplicité opérationnelle. Ses coefficients expliquent le rôle de variables comme l'engrais, l'irrigation, la pluviométrie et l'altitude. L'arbre de décision pourrait être conservé comme outil pédagogique, mais pas comme modèle principal.

## Données Supplémentaires Recommandées

Pour améliorer les prédictions, il faudrait intégrer le type de sol et le pH par zone agricole, car la fertilité influence le rendement. L'historique des maladies et ravageurs par province et saison serait aussi utile. L'accès au crédit et aux intrants mesurerait mieux la capacité réelle des ménages à appliquer de bonnes pratiques. Les prix de marché devraient être ajoutés, car ils influencent les choix de culture. Enfin, un indice satellitaire comme le NDVI pourrait mesurer l'état des cultures pendant la saison.

## Limites du Modèle et Niveau de Confiance

Les résultats doivent rester prudents. Le dataset utilisé est simulé ; un déploiement réel nécessiterait une validation avec des récoltes effectivement observées sur le terrain. Le cas Gitega-Haricot à 430 mm illustre cette limite : le modèle peut produire une probabilité élevée même lorsque la situation agronomique présente un risque, parce que les données d'entraînement ne couvrent pas suffisamment ce type de cas.

Le modèle ne tient pas compte des chocs soudains comme les sécheresses extrêmes, les inondations, les conflits, les hausses rapides des prix ou les ruptures d'approvisionnement en intrants. Il ne doit donc pas être utilisé comme un système de décision autonome. Il peut servir d'outil d'aide à la décision pour orienter l'analyse et comparer des scénarios, mais ses prédictions doivent être complétées par l'expertise agronomique locale et par des données récentes de terrain.
