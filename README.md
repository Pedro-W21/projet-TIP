# projet-TIP
Repo de code pour le projet TIP de 4TC du groupe de Anaïs DAGNET, Midoli CAILLON, Fadwa MERZAK, Pierre WEISSE en 4TC2

## Comment exécuter

- installez MATLAB version >= 2020a sur votre machine
- téléchargez une distribution du dataset food-11 séparée entre des images labelisées par dossiers dans un sous-dossier "train" et des images de test non labellisées dans un sous-dossier "test"
- clonez ce repo sur votre machine si vous ne l'avez pas déjà : `git clone https://github.com/Pedro-W21/projet-TIP`
- Pour entraîner un modèle :
    - en entraînement "from scratch"
        - modifiez les paramètres d'entraînement du modèle dans le fichier `training_code.m`
        - lancez `training_code(networkFileName, unzippedPath)` dans la ligne de commande MATLAB où `networkFileName` est le nom de fichier dans lequel vous voulez sauvegarder le réseau après l'entraînement et `unzippedPath` est le chemin (relatif ou absolu) vers votre distribution du dataset food-11 décompressé (pas sous la forme d'une archive), par exemple si votre dossier de travail est le dossier `code` et que votre dataset est à la racine de votre copie locale de ce dépôt, alors `unzippedPath` peut être `../food-11` ou `/chemin/vers/ce/repo/projet-TIP/food-11`
    - en entraînement par transfert
        - modifiez les paramètres d'entraînement du modèle dans le fichier `transfer_learning.m`
        - exportez un modèle depuis `deepNetworkDesigner` dans le workspace
        - lancer `transfer_learning(networkFileName, unzippedPath, network)` dans la ligne de commande MATLAB où `unzippedPath` et `networkFileName` sont les mêmes types de données que pour l'entraînement from scratch, et `network` est le nom de variable du modèle exporté dans le workspace
- Pour tester un modèle :
    - exécutez la fonction `test_network` dans votre console MATLAB
    - exemple : `test_network("food11_network.mat", "../food-11", "test_output.json")`

## Comment changer le modèle "from scratch" et modifier la taille de l'input des modèles entraînés

- dans `create_model_layers.m`
    - changez la définition du modèle dans la fonction `create_model_layers`
- dans `ModelConstants.m`
    - changez les constantes de modèle (si nécessaire) comme la taille de l'input