%On évalue le modèle sur un sous-ensemble de validation
function [f1_macro, perClass] = evaluateModel(netOrPath)
% Évalue macro-F1 sur la validation + affiche une confusion chart.
% netOrPath : réseau (objet DAGSeriesNetwork/dlnetwork) OU chemin .mat

scriptsDir = fileparts(mfilename('fullpath'));
S1 = load(fullfile(scriptsDir,'augDatastores.mat'),'augVal');
S2 = load(fullfile(scriptsDir,'dataStores.mat'),'imdsValidation');
augVal = S1.augVal;
imdsValidation = S2.imdsValidation;

% Charger le réseau
if ischar(netOrPath) || isstring(netOrPath)
    tmp = load(netOrPath);
    fns = fieldnames(tmp);
    net = tmp.(fns{1});          % prend la 1ère variable (ex: netShort, netBest)
else
    net = netOrPath;
end

% Prédire sur la validation
[YPred, ~] = classify(net, augVal);
YTrue = imdsValidation.Labels;

% Matrice de confusion & métriques
[C, order] = confusionmat(YTrue, YPred);

prec = diag(C) ./ max(1, sum(C,1))';
rec  = diag(C) ./ max(1, sum(C,2));
f1   = 2 * (prec .* rec) ./ max(1e-12, (prec + rec));
f1_macro = mean(f1);

perClass = table(order, prec, rec, f1, ...
    'VariableNames', {'Class','Precision','Recall','F1'});

% Affichage
figure('Name','Confusion Chart');
confusionchart(YTrue, YPred);
title(sprintf('Validation — Macro-F1 = %.4f', f1_macro));

disp(perClass);
fprintf('Macro-F1 (validation) = %.4f\n', f1_macro);
end
