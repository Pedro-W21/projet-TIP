function training_code(networkFileName, unzippedPath)
    % function that performs a full training run on the model specified in create_model_layers.m with the provided food-11 dataset (technically any dataset formatted like the food-11 it expects)
    % networkFileName : the name of the file you want the network to be output to
    % unzippedPath : the path to your local food-11 dataset, relative or absolute
    try
        [augmentedTrainData, validateData, classNames, validateLabels] = load_dataset(unzippedPath);
        
        numClasses = length(classNames);
        layers = create_model_layers(numClasses);

        options = trainingOptions('adam', ...
            'MaxEpochs', 30, ...
            'MiniBatchSize', 256, ...
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
        
        [YPred, ~] = classify(net, validateData);

        % Matrice de confusion & métriques
        [C, order] = confusionmat(validateLabels, YPred);
        
        prec = diag(C) ./ max(1, sum(C,1))';
        rec  = diag(C) ./ max(1, sum(C,2));
        f1   = 2 * (prec .* rec) ./ max(1e-12, (prec + rec));
        f1_macro = mean(f1);

        % Affichage
        figure('Name','Confusion Chart');
        confusionchart(validateLabels, YPred);
        title(sprintf('Validation - Macro-F1 = %.4f', f1_macro));

        % Save the trained network
        save(networkFileName, 'net', 'classNames');
        fprintf('\nNetwork saved\n');
        
    catch ME
        fprintf('Error: %s\n', ME.message);
    end
end

% exemple de call : training_code("food11_network.mat", "../food-11")