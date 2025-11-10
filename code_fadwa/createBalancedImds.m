% Oversampling : répliquer aléatoirement pour atteindre la taille de la plus grande classe
function balancedImds = createBalancedImds(imdsTrain)

    if nargin < 1
        error('createBalancedImds requires imdsTrain as input.');
    end

    tbl = countEachLabel(imdsTrain);
    maxCount = max(tbl.Count);

    filesBalanced = {};
    labelsBalanced = categorical();

    % Boucle sur chaque classe pour répliquer
    for i = 1:height(tbl)
        thisLabel = tbl.Label(i);
        % Récupère les fichiers de la classe courante
        files = imdsTrain.Files(imdsTrain.Labels == thisLabel);
        nFiles = numel(files);
        if nFiles == 0
            warning('Aucune image pour la classe %s — skip', string(thisLabel));
            continue;
        end
        % Indices aléatoires (avec remplacement) pour atteindre maxCount
        idx = randi(nFiles, [maxCount, 1]);
        filesRep = files(idx);
        filesBalanced = [filesBalanced; filesRep];
        labelsBalanced = [labelsBalanced; repmat(thisLabel, numel(filesRep), 1)];
    end

    % --- debug checks avant création datastore ---
    fprintf('Total filesBalanced = %d\n', numel(filesBalanced));
    if isempty(filesBalanced)
        error('filesBalanced est vide -> aucun fichier ajouté. Vérifie imdsTrain.');
    end
    % s'assurer que labelsBalanced a la même taille
    if numel(labelsBalanced) ~= numel(filesBalanced)
        error('labelsBalanced (%d) et filesBalanced (%d) n''ont pas la même taille.', ...
            numel(labelsBalanced), numel(filesBalanced));
    end

    % Mélanger les listes AVANT de créer l'imageDatastore (plus robuste)
    perm = randperm(numel(filesBalanced));
    filesBalanced = filesBalanced(perm);
    labelsBalanced = labelsBalanced(perm);

    % Créer l'imageDatastore équilibré
    balancedImds = imageDatastore(filesBalanced, 'Labels', labelsBalanced);

    % Afficher le résultat
    disp('✅ Dataset équilibré créé avec succès :');
    disp(countEachLabel(balancedImds));

    % Sauvegarder pour réutilisation
    if ~exist('scripts','dir')
        mkdir('scripts');
    end
    save(fullfile('scripts','balancedImds.mat'),'balancedImds');

end
