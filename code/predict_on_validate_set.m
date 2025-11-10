function predictions = predict_on_validate_set(networkFile, datasetPath)
    data = load(networkFile);
    net = data.net;
    classNames = data.classNames;

    testPath = fullfile(datasetPath, 'train');
    
    testSet = imageDatastore(testPath, ...
        'IncludeSubfolders', true);

    validateIndices = 1:10:length(testSet.Files);
    testSet = subset(testSet, validateIndices)
    testSet = augmentedImageDatastore(net.Layers(1).InputSize, testSet);
    testSet
    net
    try
        scores = minibatchpredict(net,testSet);
    catch ME
        scores = predict(net,testSet);
    end
    predictions = scores2label(scores,classNames);

end