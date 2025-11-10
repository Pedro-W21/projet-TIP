%On prédit les classes du dossier test et on génère le fichier JSON final à soumettre
function predictImage(imgPath)
% Usage: predictImage('chemin/vers/image.jpg')

S = load('models/mobilenetv2_best.mat','netBest','inputSize');
net = S.netBest; inputSize = S.inputSize;

I = imread(imgPath);
if size(I,3)==1, I = cat(3,I,I,I); end

% prétraitement (resize/crop au centre)
Iin = center_resize(I, inputSize);
[YPred, scores] = classify(net, Iin);

% Affichage
figure('Name','Prediction'); 
subplot(1,2,1); imshow(I); title('Original');

subplot(1,2,2);
[vals, idx] = maxk(scores,5);
labels = string(net.Layers(end).Classes(idx));
bar(vals); xticklabels(labels); xtickangle(20);
ylim([0 1]); title(sprintf('Top-5 — pred: %s (%.2f)', string(YPred), max(scores)));
end

function J = center_resize(I, outHW)
h=size(I,1); w=size(I,2); s=max(outHW./[h w]);
I2 = imresize(I, s);
y1=floor((size(I2,1)-outHW(1))/2)+1; x1=floor((size(I2,2)-outHW(2))/2)+1;
J = I2(y1:y1+outHW(1)-1, x1:x1+outHW(2)-1, :);
end
