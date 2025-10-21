# projet-TIP
Repo de code pour le projet TIP de 4TC

## Comment exécuter

- installer MATLAB version >= 2020a sur votre machine
- cloner ce repo sur votre machine
- Pour entraîner un modèle :
    - en entraînement "from scratch"
        - changer le nom de fichier de réseau `networkFileName` utilisé pour appeler la fonction `main` dans `code/training_code.m` pour choisir le nom de fichier que vous voulez comme nom de réseau
        - changer le chemin `unzippedPath` utilisé pour appeler la fonction `main` dans `code/training_code.m` pour pointer vers votre copie locale du dataset `food-11`
        - lancer `training_code.m`
    - en entraînement par transfert
        - avoir un modèle exporté depuis `deepNetworkDesigner` dans le workspace,
        - lancer `transfer_learning(networkFileName, unzippedPath, network)` dans la ligne de commande où `unzippedPath` et `networkFileName` sont les mêmes types de données que pour l'entraînement from scratch, et `network` est le nom de variable du modèle exporté dans le workspace
- Pour tester un modèle :
    - exécutez la fonction `test_network` dans votre console MATLAB
    - exemple : `test_network("food11_network.mat", "../food-11", "test_output.json")`

## Comment changer le modèle "from scratch"

- dans `training_code.m`
    - changer la définition du modèle dans `create_model_layers`
- dans `ModelConstants.m`
    - changer les constantes de modèle (si nécessaire) comme la taille de l'input