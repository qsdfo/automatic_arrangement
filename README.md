# Base de données
## Import données
- alignment ?
    - chopper score alignment sur toute la db et tester plusieurs valeurs de gapopen et gapextend. prendre la meilleure (normaliser par rapport a cette valeur)
- Re-run avec best parameters pour obtenir une base alignée
- Écrire un texte sur la database
- Unit for new beginning/end measure

## IMSLP
- Une fois les paires créées, remove les paires égales (ça arrive, les pianorolls peuvent avoir des noms différents, mais même contenu) DANS IMSLP
- Create an index for orchestration ->
        index_in_orchestration | db_path_orch-index | db_path_solo-index
   exemple :
        1093 | Kunsterfuge1023 | Musicalion384398

# Data representation
## 1
Real units

## 2
Un instrument = un pixel
2 canaux : dynamique et pitch
TODO 1 = observer sur pièces du répertoire
