function netBest = trainModel(cfg)
% trainModel.m — MobileNetV2 sur Food-11
%  - Phase 1: Warm-up (classifier only)
%  - Phase 2: Fine-tune (unfreeze all)
%  - Early stopping sur macro-F1 validation
%  - Cosine LR scheduler (maj à chaque époque)
%
% Sauvegardes:
%   models/mobilenetv2_best.mat   (meilleur net selon macro-F1)
%   models/mobilenetv2_last.mat   (dernier net entraîné)

%% -------- Config par défaut --------
def = struct( ...
    'epochsWarm', 4, ...      % 3–5
    'epochsFT',   18, ...     % 15–25
    'batch',      64, ...
    'lrWarmMax',  1e-5, ...
    'lrFTMax',    1e-4, ...
    'lrFTMin',   1e-6, ...
    'patience',   5, ...
    'plots',      false ...
);
if nargin < 1 || isempty(cfg), cfg = def; else, cfg = setDefaults(cfg, def); end

%% -------- Charger data & classes --------
scriptsDir = fileparts(mfilename('fullpath'));
S1 = load(fullfile(scriptsDir,'augDatastores.mat'),'augTrain','augVal','inputSize');
S2 = load(fullfile(scriptsDir,'dataStores.mat'),'imdsTrain','imdsValidation');

augTrain = S1.augTrain;
augVal   = S1.augVal;
inputSize = S1.inputSize; %#ok<NASGU>
imdsValidation = S2.imdsValidation;

classes = categories(S2.imdsTrain.Labels);
numClasses = numel(classes);

%% -------- Construire MobileNetV2 + classifier --------
try
    netBase = mobilenetv2;   % Add-on requis
catch
    error(['MobileNetV2 introuvable. Installe l’add-on "Deep Learning Toolbox Model for MobileNet-v2" ' ...
           'depuis Add-Ons > Get Add-Ons.']);
end
lgraph = layerGraph(netBase);

% Remplacer la dernière fullyConnected
fcIdx = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.FullyConnectedLayer'), lgraph.Layers), 1, 'last');
lgraph = replaceLayer(lgraph, lgraph.Layers(fcIdx).Name, ...
    fullyConnectedLayer(numClasses, 'Name','new_fc', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10));

% Remplacer la classification layer (sélectionne la première si plusieurs)
clIdx = find(arrayfun(@(L) isa(L,'nnet.cnn.layer.ClassificationOutputLayer'), lgraph.Layers), 1, 'first');
lgraph = replaceLayer(lgraph, lgraph.Layers(clIdx).Name, classificationLayer('Name','new_classoutput'));

%% -------- Phase 1: Warm-up (freeze backbone) --------
lgraphWarm = freezeBackbone(lgraph, {'new_fc','new_classoutput'});
bestF1 = -inf; bestNet = []; netFT = [];  % netFT init pour la sauvegarde "last"

fprintf('\n==> Phase 1: Warm-up (classifier only), %d époques\n', cfg.epochsWarm);
waited = 0;
for ep = 1:cfg.epochsWarm
    lr = cosineLRmin(cfg.lrWarmMax, cfg.lrWarmMin, ep, cfg.epochsWarm);
    opts = trainingOptions('adam', ...
        'InitialLearnRate', lr, ...
        'MiniBatchSize', cfg.batch, ...
        'MaxEpochs', 1, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augVal, ...
        'ValidationFrequency', max(1, floor(numel(augTrain.Files)/cfg.batch/4)), ...
        'ExecutionEnvironment','auto', ...
        'Verbose', true, ...
        'Plots', ternary(cfg.plots,'training-progress','none'));
    try, gpuDevice(1); reset(gpuDevice); end %#ok<TRYNC>
    netWarm = trainNetwork(augTrain, lgraphWarm, opts);

    % Eval macro-F1
    f1 = macroF1(netWarm, augVal, imdsValidation.Labels);
    fprintf('Warm-up Epoch %02d/%02d | LR=%.2e | macro-F1=%.4f\n', ep, cfg.epochsWarm, lr, f1);

    if f1 > bestF1
        bestF1 = f1; bestNet = netWarm; waited = 0;
    else
        waited = waited + 1;
        if waited >= cfg.patience
            fprintf('Early-stop warm-up (patience atteinte).\n');
            break;
        end
    end

    % Continuer à partir du réseau entraîné, backbone toujours gelé
    lgraphWarm = layerGraph(netWarm);
    lgraphWarm = freezeBackbone(lgraphWarm, {'new_fc','new_classoutput'});
end

%% -------- Phase 2: Fine-tune (unfreeze all) --------
fprintf('\n==> Phase 2: Fine-tune (full), %d époques\n', cfg.epochsFT);
lgraphFT = layerGraph(bestNet);
lgraphFT = unfreezeBackbone(lgraphFT, 'new_fc');

waited = 0;
for ep = 1:cfg.epochsFT
    lr = cosineLRmin(cfg.lrFTMax, cfg.lrFTMin, ep, cfg.epochsFT);
    opts = trainingOptions('adam', ...
        'InitialLearnRate', lr, ...
        'MiniBatchSize', cfg.batch, ...
        'MaxEpochs', 1, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', augVal, ...
        'ValidationFrequency', max(1, floor(numel(augTrain.Files)/cfg.batch/4)), ...
        'ExecutionEnvironment','auto', ...
        'Verbose', true, ...
        'Plots', ternary(cfg.plots,'training-progress','none'));
    try, gpuDevice(1); reset(gpuDevice); end %#ok<TRYNC>
    netFT = trainNetwork(augTrain, lgraphFT, opts);

    f1 = macroF1(netFT, augVal, imdsValidation.Labels);
    fprintf('Fine-tune Epoch %02d/%02d | LR=%.2e | macro-F1=%.4f (best=%.4f)\n', ep, cfg.epochsFT, lr, f1, bestF1);

    if f1 > bestF1
        bestF1 = f1; bestNet = netFT; waited = 0;
        fprintf('  ↳ Nouveau meilleur checkpoint ✅\n');
    else
        waited = waited + 1;
        if waited >= cfg.patience
            fprintf('Early-stop fine-tune (patience atteinte).\n');
            break;
        end
    end

    lgraphFT = layerGraph(netFT);
    lgraphFT = unfreezeBackbone(lgraphFT, 'new_fc');
end

%% -------- Sauvegardes --------
if ~isfolder('models'), mkdir('models'); end
netBest = bestNet;
if isempty(netFT), netFT = bestNet; end  % si fine-tune interrompu avant 1er ep
save('models/mobilenetv2_best.mat','netBest','classes','inputSize','-v7.3');
save('models/mobilenetv2_last.mat','netFT','classes','inputSize','-v7.3');
fprintf('\n✅ Meilleur modèle sauvegardé: models/mobilenetv2_best.mat (macro-F1=%.4f)\n', bestF1);

end

%% ================= Helpers =================
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end

function lgraph = freezeBackbone(lgraph, keepNames)
% Gèle toutes les couches sauf celles listées
keepNames = cellstr(string(keepNames));
layers = lgraph.Layers;
for i = 1:numel(layers)
    L = layers(i);
    if ~any(strcmp(string(L.Name), string(keepNames)))
        if isprop(L,'WeightLearnRateFactor'), L.WeightLearnRateFactor = 0; end
        if isprop(L,'BiasLearnRateFactor'),   L.BiasLearnRateFactor   = 0; end
    end
    layers(i) = L;
end
lgraph = rebuild(layers, lgraph.Connections);
end

function lgraph = unfreezeBackbone(lgraph, headName)
% Dégèle toutes les couches, met la tête à LR factor plus élevé
layers = lgraph.Layers;
for i = 1:numel(layers)
    L = layers(i);
    if isprop(L,'WeightLearnRateFactor')
        if strcmp(string(L.Name), string(headName))
            L.WeightLearnRateFactor = 10; L.BiasLearnRateFactor = 10;
        else
            L.WeightLearnRateFactor = 1;  L.BiasLearnRateFactor = 1;
        end
    end
    layers(i) = L;
end
lgraph = rebuild(layers, lgraph.Connections);
end

function lgraph = rebuild(layers, connections)
% Reconstruit un layerGraph et reconnecte ligne par ligne
lgraph = layerGraph();
for i = 1:numel(layers)
    lgraph = addLayers(lgraph, layers(i));
end
if isempty(connections) || height(connections) == 0
    return;
end
src = connections.Source; dst = connections.Destination;
if isstring(src), src = cellstr(src); end
if isstring(dst), dst = cellstr(dst); end
for k = 1:numel(src)
    lgraph = connectLayers(lgraph, src{k}, dst{k});
end
end

function lr = cosineLRmin(lrMax, lrMin, t, T)
% Cosine annealing avec plancher (lrMin > 0)
% t = 1..T
alpha = 0.5 * (1 + cos(pi * (t-1) / (T-1 + eps)));
lr = lrMin + (lrMax - lrMin) * alpha;
end


function f1_macro = macroF1(net, augVal, YTrue)
% Calcule macro-F1 sur validation
[YPred, ~] = classify(net, augVal);
YPred = gather(YPred);
YTrue = gather(YTrue);
[C, ~] = confusionmat(YTrue, YPred);
prec = diag(C) ./ max(1, sum(C,1))';
rec  = diag(C) ./ max(1, sum(C,2));
f1   = 2 * (prec .* rec) ./ max(1e-12, (prec + rec));
f1_macro = mean(f1);
end

function cfg = setDefaults(cfg, def)
fn = fieldnames(def);
for i = 1:numel(fn)
    k = fn{i};
    if ~isfield(cfg,k) || isempty(cfg.(k))
        cfg.(k) = def.(k);
    end
end
end
