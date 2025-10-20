function [augmentedTrainData, resizedValidate, classNames] = load_dataset(unzippedPath)
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

    resizedValidate = augmentedImageDatastore(ModelConstants.imgSize, validateData);

    allIndices = 1:1:length(allData.Files);
    
    trainIndices = setdiff(allIndices, validateIndices);
    trainData = subset(allData, trainIndices);


    augmenter = imageDataAugmenter( ...
        "RandRotation", [-10, 10], ...
        "RandXTranslation", [-5, 5], ...
        "RandYTranslation", [-5, 5], ...
        "RandXReflection",true ...
    );

    augmentedTrainData = augmentedImageDatastore(ModelConstants.imgSize, trainData, "DataAugmentation", augmenter);

    classNames = unique(trainData.Labels);
    
    fprintf('Dataset loaded successfully!\n');
    fprintf('Number of classes: %d\n', length(classNames));
    fprintf('Total training images: %d\n', length(trainData.Files));
    fprintf('Classes: {%s}\n', strjoin(string(classNames), ', '));
    

    classCounts = countcats(trainData.Labels);
    fprintf('\nClass distribution:\n');
    for i = 1:length(classNames)
        fprintf('  %s: %d images\n', classNames(i), classCounts(i));
    end
end

