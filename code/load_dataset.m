function [augmentedTrainData, resizedValidate, classNames, validateLabels] = load_dataset(unzippedPath)
    % Load and extract the food-11 dataset archive
    % Input: 
    % - archivePath : path to the .zip/.tar archive file
    % Output: 
    % - trainData : imageDatastore object ready for CNN training
    % - classNames : cell array of class names
   
    
    trainPath = fullfile(unzippedPath, 'train');
    
    % Create imageDatastore from the train folder structure
    % The folder structure is assumed to be:
    % train/
    %   class1/
    %     image1.jpg
    %     image2.jpg
    %   class2/
    %     image3.jpg
    %     image4.jpg
    %   ...

    
    allData = imageDatastore(trainPath, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');

    validateIndices = 1:10:length(allData.Files);
    validateData = subset(allData, validateIndices);

    validateLabels = validateData.Labels;

    resizedValidate = augmentedImageDatastore(ModelConstants.imgSize, validateData);

    allIndices = 1:1:length(allData.Files);
    
    trainIndices = setdiff(allIndices, validateIndices);
    trainData = subset(allData, trainIndices);

    tbl = countEachLabel(trainData);
    maxCount = max(tbl.Count);

    filesBalanced = {};
    labelsBalanced = categorical();

    % Boucle sur chaque classe pour répliquer
    for i = 1:height(tbl)
        thisLabel = tbl.Label(i);
        
        % Récupère les fichiers de la classe courante
        files = trainData.Files(trainData.Labels == thisLabel);
        nFiles = numel(files);

        % Indices aléatoires (avec remplacement) pour atteindre maxCount
        idx = randi(nFiles, [maxCount, 1]);
        filesRep = files(idx);
        filesBalanced = [filesBalanced; filesRep];
        labelsBalanced = [labelsBalanced; repmat(thisLabel, numel(filesRep), 1)];
    end

    
    balancedImds = imageDatastore(filesBalanced, 'Labels', labelsBalanced);

    augmenter = imageDataAugmenter( ...
        "RandRotation", [-10, 10], ...
        "RandXTranslation", [-5, 5], ...
        "RandYTranslation", [-5, 5], ...
        "RandXReflection",true ...
    );

    augmentedTrainData = augmentedImageDatastore(ModelConstants.imgSize, balancedImds, "DataAugmentation", augmenter);

    classNames = unique(balancedImds.Labels);
    
    fprintf('Dataset loaded successfully!\n');
    fprintf('Number of classes: %d\n', length(classNames));
    fprintf('Total training images: %d\n', length(balancedImds.Files));
    fprintf('Classes: {%s}\n', strjoin(string(classNames), ', '));
    

    classCounts = countcats(balancedImds.Labels);
    fprintf('\nClass distribution:\n');
    for i = 1:length(classNames)
        fprintf('  %s: %d images\n', classNames(i), classCounts(i));
    end
end

