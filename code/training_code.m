clear all;
close all;

main("food11_network.mat", "../food-11")

function cnnLayers = create_model_layers(numClasses)
    % Create the layers of a CNN for classification according to numClasses
    % Input: 
    % - numClasses : number of food classes
    % Output:
    % - cnnLayers : layer array for the CNN
    
    layers = [
        % Input layer
        imageInputLayer(ModelConstants.imgSize, 'Name', 'input')
        
        % Convolution 1 
        convolution2dLayer(5, 64, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        % Convolution 2 
        convolution2dLayer(5, 128, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        
        % Convolution 3
        convolution2dLayer(5, 256, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
        
        % FC1
        fullyConnectedLayer(512, 'Name', 'fc1')
        batchNormalizationLayer('Name', 'bn_fc1')
        reluLayer('Name', 'relu_fc1')

        % FC2
        fullyConnectedLayer(256, 'Name', 'fc2')
        batchNormalizationLayer('Name', 'bn_fc2')
        reluLayer('Name', 'relu_fc2')

        dropoutLayer(0.5, 'Name', 'dropout')
        
        fullyConnectedLayer(numClasses, 'Name', 'fc3')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classoutput')
    ];
    
    cnnLayers = layers;
end

function main(networkFileName, unzippedPath)
    
    try
        [augmentedTrainData, validateData, classNames] = load_dataset(unzippedPath);
        
        numClasses = length(classNames);
        layers = create_model_layers(numClasses);

        options = trainingOptions('adam', ...
            'MaxEpochs', 20, ...
            'MiniBatchSize', 500, ...
            'InitialLearnRate', 0.01, ...
            "LearnRateSchedule", "piecewise", ...
            "LearnRateDropFactor", 0.3, ...
            "LearnRateDropPeriod", 3, ...
            "L2Regularization", 0.001, ...
            'Shuffle', 'every-epoch', ...
            "ValidationData", validateData, ...
            'ValidationFrequency', 40, ...
            'Verbose', true, ...
            'Plots', 'training-progress');
        
        % Train the network
        fprintf('\nStarting training...\n');
        net = trainNetwork(augmentedTrainData, layers, options);
        
        % Save the trained network
        save(networkFileName, 'net', 'classNames');
        fprintf('\nNetwork saved as food11_cnn_network.mat\n');
        
    catch ME
        fprintf('Error: %s\n', ME.message);
    end
end
