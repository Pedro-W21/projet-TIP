%On prépare le dataset et on crée les imageDatastore
% Définir les chemins
trainDir = fullfile('dataset', 'train');
testDir  = fullfile('dataset', 'test');

% Charger les ensembles d’images
imdsTrain = imageDatastore(trainDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

imdsTest = imageDatastore(testDir, ...
    'IncludeSubfolders', false);  % test set n’a PAS de labels !

disp("✅ Classes trouvées dans le dossier train :");
disp(categories(imdsTrain.Labels));

[imdsTrain, imdsValidation] = splitEachLabel(imdsTrain, 0.8, 'randomized');

disp("✅ Dataset chargé avec succès !");
disp("Nombre d’images d’entraînement : " + numel(imdsTrain.Files));
disp("Nombre d’images de validation : " + numel(imdsValidation.Files));
disp("Nombre d’images de test : " + numel(imdsTest.Files));

save('scripts/dataStores.mat', 'imdsTrain', 'imdsValidation', 'imdsTest');
