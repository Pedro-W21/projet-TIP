% prepareAugDatastores.m
% Prépare augTrain / augVal pour MobileNetV2 et les sauvegarde à côté du script.

%% Localisation du dossier du script (ex: .../scripts)
scriptsDir = fileparts(mfilename('fullpath'));

%% Charger balancedImds (si pas en workspace)
if ~exist('balancedImds','var')
    fBalanced = fullfile(scriptsDir,'balancedImds.mat');
    if exist(fBalanced,'file')
        S = load(fBalanced,'balancedImds');
        balancedImds = S.balancedImds;
    else
        error(['balancedImds introuvable. ' ...
               'Exécute createBalancedImds(imdsTrain) d''abord (balancedImds.mat attendu dans scripts).']);
    end
end

%% Charger imdsValidation (si pas en workspace)
if ~exist('imdsValidation','var')
    fData = fullfile(scriptsDir,'dataStores.mat');
    if exist(fData,'file')
        S = load(fData,'imdsValidation');
        imdsValidation = S.imdsValidation;
    else
        error(['imdsValidation introuvable. ' ...
               'Exécute loadData.m d''abord (dataStores.mat attendu dans scripts).']);
    end
end

%% Récupérer la taille d'entrée de MobileNetV2
try
    net = mobilenetv2;                             % nécessite l’add-on MobileNetV2
catch
    error(['MobileNetV2 indisponible. Installe l’add-on "Deep Learning Toolbox Model for MobileNet-v2" ' ...
           'depuis Add-Ons > Get Add-Ons.']);
end
inputSize = net.Layers(1).InputSize(1:2);         % généralement [224 224]

%% Définir les augmentations (modifiables)
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-20 20], ...
    'RandXReflection',true, ...
    'RandScale',[0.9 1.1], ...
    'RandXTranslation',[-10 10], ...
    'RandYTranslation',[-10 10]);

%% Créer les augmentedImageDatastore
augTrain = augmentedImageDatastore(inputSize, balancedImds, ...
    'DataAugmentation', imageAugmenter);

augVal   = augmentedImageDatastore(inputSize, imdsValidation);

% (optionnel) batch size par défaut pour l'entraînement
try
    augTrain.MiniBatchSize = 128;
    augVal.MiniBatchSize   = 128;
catch
    % certaines versions ne permettent pas l’assignation ici — sans gravité
end

%% Sauvegarde à côté du script (pas de 'scripts/scripts')
save(fullfile(scriptsDir,'augDatastores.mat'), 'augTrain','augVal','inputSize','-v7.3');

%% Logs
disp('✅ Augmented datastores prêts :');
disp([' - Train (balanced) : ' num2str(numel(balancedImds.Files)) ' images']);
disp([' - Val (original)   : ' num2str(numel(imdsValidation.Files)) ' images']);
disp([' - InputSize        : [' num2str(inputSize) ']']);
