function netShort = trainShort()
% charge augTrain/augVal + classes
scriptsDir = fileparts(mfilename('fullpath'));
S1 = load(fullfile(scriptsDir,'augDatastores.mat'), 'augTrain','augVal','inputSize');
S2 = load(fullfile(scriptsDir,'dataStores.mat'), 'imdsTrain');

augTrain = S1.augTrain;
augVal   = S1.augVal;
classes  = categories(S2.imdsTrain.Labels);
numClasses = numel(classes);

% modèle de base
net = mobilenetv2;
lgraph = layerGraph(net);

% remplace la dernière FC
fcIdx = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.FullyConnectedLayer'), lgraph.Layers),1,'last');
lgraph = replaceLayer(lgraph, lgraph.Layers(fcIdx).Name, ...
    fullyConnectedLayer(numClasses, 'Name','new_fc', ...
        'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10));

% remplace la classification layer
clIdx = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.ClassificationOutputLayer'), lgraph.Layers));
lgraph = replaceLayer(lgraph, lgraph.Layers(clIdx).Name, ...
    classificationLayer('Name','new_classoutput'));

% options (rapides)
opts = trainingOptions('adam', ...
    'InitialLearnRate',5e-4, ...
    'MiniBatchSize',64, ...
    'MaxEpochs',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augVal, ...
    'ExecutionEnvironment','auto', ...
    'Verbose',true, ...
    'Plots','training-progress');

% (facultatif) sélection GPU & reset mémoire
try, gpuDevice(1); reset(gpuDevice); end

netShort = trainNetwork(augTrain, lgraph, opts);

% sauvegarde légère
if ~isfolder('models'), mkdir('models'); end
save(fullfile('models','mobilenetv2_short.mat'),'netShort','classes','-v7.3');
end
