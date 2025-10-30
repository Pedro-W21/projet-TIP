# Model 

a LeNet-5 inspired model, with more layers and parameters to try to make it better at higher resolutions (128x128x3)

# Layers 

layers = [
    % Input layer
    imageInputLayer(ModelConstants.imgSize, 'Name', 'input')
    
    % Convolution 1 
    convolution2dLayer(48, 12, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    % Convolution 2 
    convolution2dLayer(16, 32, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')

    % Convolution 3 
    convolution2dLayer(8, 64, 'Padding', 'same', 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
    
    % FC1
    fullyConnectedLayer(250, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn_fc1')
    reluLayer('Name', 'relu_fc1')
    
    % FC1
    fullyConnectedLayer(120, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn_fc2')
    reluLayer('Name', 'relu_fc2')

    % FC2
    fullyConnectedLayer(84, 'Name', 'fc3')
    batchNormalizationLayer('Name', 'bn_fc3')
    reluLayer('Name', 'relu_fc3')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc4')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];