function test_network(networkFile, datasetPath, testFileName)
    % Tests the given network on the test dataset and outputs the relevant .json file 
    % Input :
    % - networkFile : path to the network file you want to test
    % - datasetPath : path to your food-11 dataset
    % - testFileName : name of the JSON file for the final test run
    data = load(networkFile);
    net = data.net;
    classNames = data.classNames;

    testPath = fullfile(datasetPath, 'test');
    testSet = imageDatastore(testPath, ...
        'IncludeSubfolders', true);
    testSet = augmentedImageDatastore(net.Layers(1).InputSize, testSet);
    
    try
        scores = minibatchpredict(net,testSet);
    catch ME
        scores = predict(net,testSet);
    end
    predictions = scores2label(scores,classNames);
    encoded = jsonencode(containers.Map(string(transpose(0:(length(predictions) - 1))),string(predictions)));

    fid = fopen(testFileName,'w');
    fprintf(fid,'%s',encoded);
    fclose(fid);
end