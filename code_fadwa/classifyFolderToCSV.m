function classifyFolderToCSV(folder, outCSV)
% Usage: classifyFolderToCSV('my_photos', 'results/preds.csv')
if nargin<2, outCSV='results/preds.csv'; end
if ~isfolder('results'), mkdir('results'); end

S = load('models/mobilenetv2_best.mat','netBest','inputSize'); 
net = S.netBest; inputSize = S.inputSize;

imds = imageDatastore(folder,'IncludeSubfolders',false);
files = imds.Files; K = numel(files);
pred = strings(K,1); conf = zeros(K,1);

for i=1:K
    I = imread(files{i}); if size(I,3)==1, I=cat(3,I,I,I); end
    Iin = center_resize(I, inputSize);
    [YP, scores] = classify(net, Iin);
    [mx,~] = max(scores);
    pred(i) = string(YP); conf(i) = mx;
end

T = table(files, pred, conf, 'VariableNames', {'file','pred','confidence'});
writetable(T, outCSV);
fprintf('✅ Écrit: %s (%d lignes)\n', outCSV, K);
end

function J = center_resize(I, outHW)
h=size(I,1); w=size(I,2); s=max(outHW./[h w]);
I2 = imresize(I, s);
y1=floor((size(I2,1)-outHW(1))/2)+1; x1=floor((size(I2,2)-outHW(2))/2)+1;
J = I2(y1:y1+outHW(1)-1, x1:x1+outHW(2)-1, :);
end
