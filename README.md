# LOP

Live Orchestral Project

## To do
- Ré-écrire le dictionnaire, avec clarinette basse, basson basse... (cf Moussorgsky)
- Insérer un formulaire dans la visualisation pour facilement passer d'un CSV à un autre
		- D'ailleurs disp_pr sera inutile dans ce cas
		- on pourra directement browser le répertoire CSV
- Copier config atom sur GPU
- Checker DB DD-Aciditeam

### Evaluation :
- Adapter les RNN
        - Label Gated sur un RNN-RBM
- Trouver une méthode d'évaluation plus adaptée
        - 1/ Harmonic respect : comparer pitch-class piano et pitch-class orchestraux
        - 2/ Dynamic comparison : loudness piano VS loudness Orchestra
        - 3/ Accuracy, Precision and Recall measure
- Focus learning
                - CF carnet notes. Idées basique, n'apprendre qu'un instrument à un moment donné.
- Measure = p(v(t:t+N)|v(t-N:t-1))
- Ré-implémenter avec X-test, and validating sets
        - Event-level
                - CRBM
                - FGCRBM
                - repeat
                - random
        - Frame-level
                - CRBM
                - FGCRBM
                - repeat
                - random
- Faire un set-up de test pour les CRBM :
        - Sur du training un peu bidon (MNIST), et sans supervised training à la fin
        - Tester différents sampling : mean-field values or sampled binary values ?
### Visualization
        - Créer des exemples bidons :
                - Telle ligne mélodique simple = ??
                - séquence de chords ??

### Pre-processing
- braids en pre-pro
        - Lire braids
        - Ecrire un résumé dans state of the art
        - Coder
        - Tester
- Word-embedding or skip-thought vectors
