# Base de données
## Import données
- Alignment
    - Compare :
        - DTW
        - Needleman-Munch
- Check random (centaine de pièces)
- Finir build_data

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
