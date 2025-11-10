function transfer_learning(networkFileName, unzippedPath, network) 
    try
        [augmentedTrainData, validateData, classNames, validateLabels] = load_dataset(unzippedPath);

        options = trainingOptions('adam', ...
            'MaxEpochs', 30, ...
            'MiniBatchSize', 30, ...
            'InitialLearnRate', 0.001, ...
            "LearnRateSchedule", "piecewise", ...
            "LearnRateDropFactor", 0.15, ...
            "LearnRateDropPeriod", 1, ...
            "L2Regularization", 0.003, ...
            'Shuffle', 'every-epoch', ...
            "ValidationData", validateData, ...
            'ValidationFrequency', 30, ...
            'Verbose', true, ...
            "Metrics", "fscore", ...
            'Plots', 'training-progress');
        
        % Train the network
        fprintf('\nStarting training...\n');
        net = trainnet(augmentedTrainData, network, "crossentropy", options);
        

        % Save the trained network
        save(networkFileName, 'net', 'classNames');
        fprintf('\nNetwork saved as %s\n', networkFileName);

        predictions = predict_on_validate_set(networkFileName, "../food-11")

        % Matrice de confusion & métriques
        [C, order] = confusionmat(validateLabels, predictions);
        
        prec = diag(C) ./ max(1, sum(C,1))';
        rec  = diag(C) ./ max(1, sum(C,2));
        f1   = 2 * (prec .* rec) ./ max(1e-12, (prec + rec));
        f1_macro = mean(f1);

        % Affichage
        figure('Name','Confusion Chart');
        confusionchart(validateLabels, predictions);
        title(sprintf('Validation - Macro-F1 = %.4f', f1_macro));

        
    catch ME
        fprintf('Error: %s\n', ME.message);
    end

end

function dlX = augmentedImageDatastore2dlarray(augimds)
    reset(augimds);
    batches = {};
    sizes = [];
    i = 0;
    while hasdata(augimds)
        tbl = read(augimds);
        size(tbl)
        size(tbl.input)
        i = i + 1;
        sizes(i,:) = size(tbl.input);
        batches{i} = tbl.input;
    end
    X = cat(4, batches{:});
    dlX = dlarray(X, 'SSCB');
end

% train_provided("food11_network_efficientnet_2.mat", "../food-11", net_1)