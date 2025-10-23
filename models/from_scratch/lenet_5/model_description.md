# Model 

LeNet-5, as described on the TIP documentation

changes : output to 11 classes, input is RGB

# Layers 

layers = [
    % Input layer
    imageInputLayer(ModelConstants.imgSize, 'Name', 'input')
    
    % Convolution 1 
    convolution2dLayer(24, 6, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    
    % Convolution 2 
    convolution2dLayer(8, 16, 'Padding', 'same', 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    
    % FC1
    fullyConnectedLayer(120, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn_fc1')
    reluLayer('Name', 'relu_fc1')

    % FC2
    fullyConnectedLayer(84, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn_fc2')
    reluLayer('Name', 'relu_fc2')
    
    fullyConnectedLayer(numClasses, 'Name', 'fc3')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')
];