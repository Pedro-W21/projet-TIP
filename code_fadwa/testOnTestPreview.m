function T = testOnTestPreview(maxShowPerBucket)
% Audit qualitatif du modèle sur dataset/test (sans labels)
% - Prédit toutes les images de test
% - Résume les confiances (histogramme, stats)
% - Montre top-N images les + confiantes et les - confiantes
% - Affiche la distribution des classes prédites
% - Retourne un tableau (file, pred, confidence)
%
% Usage:
%   T = testOnTestPreview;             % N=20 par défaut
%   T = testOnTestPreview(24);         % affiche 24 images / bucket
%
% Aucune écriture de JSON.

if nargin<1, maxShowPerBucket = 20; end

% --- Charger modèle ---
S = load('models/mobilenetv2_best.mat','netBest','inputSize');
net = S.netBest; inputSize = S.inputSize;
classes = string(net.Layers(end).Classes);

% --- Charger test set (tri par nom numérique si possible) ---
root = projectRoot();
imdsTest = imageDatastore(fullfile(root,'dataset','test'), 'IncludeSubfolders', false);
[~, baseNames, ~] = cellfun(@fileparts, imdsTest.Files, 'UniformOutput', false);
nums = str2double(baseNames); [~, ord] = sortrows([isnan(nums), nums],[1 2]);
imdsTest.Files = imdsTest.Files(ord); baseNames = baseNames(ord);

K = numel(imdsTest.Files);
pred = strings(K,1); conf = zeros(K,1);

% --- Prédictions ---
wb = waitbar(0, 'Prédiction sur test...');
for i=1:K
    I = imread(imdsTest.Files{i});
    if size(I,3)==1, I = cat(3,I,I,I); end
    Iin = center_resize(I, inputSize);
    [yp, scores] = classify(net, Iin);
    [mx, ~] = max(scores);
    pred(i) = string(yp);
    conf(i) = mx;
    if mod(i,50)==0 || i==K, waitbar(i/K, wb); end
end
close(wb);

% --- Tableau résultat (en mémoire) ---
T = table(imdsTest.Files, baseNames, pred, conf, 'VariableNames', {'file','id','pred','confidence'});
disp(head(T,10));

% --- Stats de confiance ---
fprintf('\nConfiance (softmax max): mean=%.3f | median=%.3f | p10=%.3f | p90=%.3f\n', ...
    mean(conf), median(conf), prctile(conf,10), prctile(conf,90));

figure('Name','Histogramme des confiances');
histogram(conf, 20); xlabel('Confiance'); ylabel('Nb images'); title('Distribution des confiances (test)');

% --- Distribution des classes prédites ---
[pCounts, gNames] = groupcounts(pred);
[~, ii] = sort(pCounts, 'descend');
figure('Name','Répartition des classes prédites');
bar(categorical(gNames(ii)), pCounts(ii));
title('Répartition des classes prédites (test)'); ylabel('Nb images');
xtickangle(30);

% --- Visualiser images les PLUS confiantes ---
[~, idxDesc] = sort(conf, 'descend');
showN = min(maxShowPerBucket, numel(idxDesc));
figure('Name','Top prédictions les PLUS confiantes'); 
tiledlayout(ceil(sqrt(showN)), ceil(sqrt(showN)), 'Padding','compact','TileSpacing','compact');
for k=1:showN
    i = idxDesc(k);
    I = imread(T.file{i});
    nexttile; imshow(I);
    title(sprintf('%s (%.2f)', T.pred(i), T.confidence(i)), 'Interpreter','none');
end

% --- Visualiser images les MOINS confiantes ---
[~, idxAsc] = sort(conf, 'ascend');
showN = min(maxShowPerBucket, numel(idxAsc));
figure('Name','Top prédictions les MOINS confiantes'); 
tiledlayout(ceil(sqrt(showN)), ceil(sqrt(showN)), 'Padding','compact','TileSpacing','compact');
for k=1:showN
    i = idxAsc(k);
    I = imread(T.file{i});
    nexttile; imshow(I);
    title(sprintf('%s (%.2f)', T.pred(i), T.confidence(i)), 'Interpreter','none', 'Color',[0.7 0 0]);
end

% --- Exemple: aperçu aléatoire 25 images ---
randIdx = randperm(K, min(25,K));
figure('Name','Aperçu aléatoire (pred + conf)'); 
tiledlayout(5,5,'Padding','compact','TileSpacing','compact');
for k=1:numel(randIdx)
    i = randIdx(k);
    I = imread(T.file{i});
    nexttile; imshow(I);
    title(sprintf('%s (%.2f)', T.pred(i), T.confidence(i)), 'Interpreter','none');
end

fprintf('\n✅ Test preview terminé. (Aucun fichier JSON généré.)\n');

end

% --------- Helpers ---------
function root = projectRoot()
here = pwd; if endsWith(here, [filesep 'scripts']), root = fileparts(here); else, root = here; end
end

function J = center_resize(I, outHW)
h=size(I,1); w=size(I,2); s=max(outHW./[h w]);
I2 = imresize(I, s);
y1=floor((size(I2,1)-outHW(1))/2)+1; x1=floor((size(I2,2)-outHW(2))/2)+1;
J = I2(y1:y1+outHW(1)-1, x1:x1+outHW(2)-1, :);
end
