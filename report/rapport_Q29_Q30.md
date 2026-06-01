# Rapport : Déploiement d'un Modèle de Prédiction Agricole au Burundi

## Analyse du Scénario 3 — Gitega, Haricot, Pluviométrie 430mm (Q29)

Le scénario 3 correspond à une parcelle de haricot située à Gitega, à 1 720 m d'altitude, avec une pluviométrie de 430 mm, une température moyenne de 18,2 °C, sans engrais et sans irrigation. Après transformation avec le même pipeline que celui utilisé à l'entraînement, les trois modèles prédisent une bonne récolte. L'arbre de décision donne une probabilité d'environ 97,9 %, la forêt aléatoire environ 86,0 %, et la régression logistique environ 90,0 %.

Ce résultat doit être interprété avec prudence. Sur le plan agronomique, le haricot demande généralement entre 400 et 600 mm d'eau pendant son cycle. Une pluviométrie de 430 mm se situe donc près de la limite basse. Elle ne signifie pas automatiquement une mauvaise récolte, mais elle indique un risque réel de stress hydrique, surtout si les pluies sont mal réparties. L'absence d'irrigation et d'engrais renforce ce risque. Les modèles prédisent pourtant une bonne récolte, probablement parce que d'autres facteurs appris dans les données, comme l'altitude de Gitega ou les relations propres à cette culture, compensent partiellement le déficit hydrique.

Il ne faut donc pas conclure que le risque est nul. La forêt aléatoire fournit la probabilité la plus faible des trois modèles pour ce scénario, ce qui montre une incertitude relative. La prédiction observée est favorable, mais elle doit être confrontée à l'état réel de la parcelle, au calendrier des pluies et aux pratiques culturales.

Trois recommandations pratiques peuvent être proposées. Premièrement, si l'eau est accessible, une irrigation complémentaire devrait être envisagée pendant les phases sensibles de floraison et de formation des gousses. Deuxièmement, l'agriculteur pourrait utiliser des variétés de haricot à cycle court, mieux adaptées au stress hydrique. Troisièmement, des techniques de conservation de l'humidité du sol, comme le paillage, le travail minimum du sol et la couverture végétale, peuvent limiter l'évaporation et améliorer la disponibilité de l'eau.

## Recommandation de Modèle pour Déploiement en Production (Q30)

La comparaison des modèles montre que la régression logistique obtient les meilleurs résultats globaux dans cette étude. D'après `metrics.json`, elle atteint une accuracy de 0,9399, un F1-score pondéré de 0,9203 et une AUC de 0,8358. Ces valeurs sont supérieures à celles de la forêt aléatoire pour le F1-score et l'AUC, et supérieures à celles de l'arbre de décision pour l'accuracy et l'AUC. En validation croisée à 5 folds, elle obtient environ 0,9373 ± 0,0085, ce qui indique une performance stable.

La forêt aléatoire obtient une accuracy de 0,9304, un F1-score de 0,8999 et une AUC de 0,7419. Sa validation croisée est très stable, avec environ 0,9341 ± 0,0040, mais son AUC plus faible indique une capacité de discrimination moins bonne. L'arbre de décision obtient une accuracy de 0,8987, un F1-score de 0,9009 et une AUC de 0,7520, avec une validation croisée d'environ 0,9254 ± 0,0141. Il est plus facile à expliquer, mais il est moins performant et plus sensible au choix de la profondeur.

Sur la base de ces résultats, je recommanderais de déployer la régression logistique comme modèle principal. Elle présente le meilleur compromis entre performance, interprétabilité et simplicité opérationnelle. Ses coefficients permettent aussi d'expliquer le rôle de variables comme l'engrais, l'irrigation, la pluviométrie et l'altitude. L'arbre de décision pourrait être conservé comme outil pédagogique d'explication, mais pas comme modèle principal.

## Données Supplémentaires Recommandées

Pour améliorer les prédictions, il faudrait intégrer le type de sol et le pH par zone agricole, car la fertilité influence directement le rendement. L'historique des maladies et ravageurs par province et par saison serait aussi utile pour représenter les pertes non climatiques. L'accès au crédit et aux intrants agricoles permettrait de mieux mesurer la capacité réelle des ménages à appliquer de bonnes pratiques. Les prix de marché devraient être ajoutés, car ils influencent les choix de culture et les investissements. Enfin, un indice satellitaire de végétation comme le NDVI pourrait fournir une mesure objective de l'état des cultures pendant la saison.

## Limites du Modèle et Niveau de Confiance

Les résultats doivent rester prudents. Le dataset utilisé est simulé ; un déploiement réel nécessiterait une validation avec des récoltes effectivement observées sur le terrain. Le nombre d'exemples par combinaison province-culture peut aussi être limité, ce qui augmente la variance des estimations locales.

Le modèle ne tient pas compte des chocs soudains comme les sécheresses extrêmes, les inondations, les conflits, les hausses rapides des prix ou les ruptures d'approvisionnement en intrants. Il ne doit donc pas être utilisé comme un système de décision autonome. Il peut servir d'outil d'aide à la décision pour orienter l'analyse et comparer des scénarios, mais ses prédictions doivent être complétées par l'expertise agronomique locale et par des données récentes de terrain.
