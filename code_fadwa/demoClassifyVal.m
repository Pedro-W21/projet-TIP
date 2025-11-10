function demoClassifyVal(N)
% Affiche N images de validation avec vrai/pred (couleur = correct/incorrect)
if nargin<1, N=16; end
S1 = load(fullfile('scripts','dataStores.mat'),'imdsValidation');
S2 = load('models/mobilenetv2_best.mat','netBest','inputSize');
imdsVal = S1.imdsValidation; net = S2.netBest; inputSize=S2.inputSize;

idx = randperm(numel(imdsVal.Files), min(N,numel(imdsVal.Files)));
tiledlayout(ceil(sqrt(N)), ceil(sqrt(N)),'Padding','compact','TileSpacing','compact');

for k=1:numel(idx)
    I = readimage(imdsVal, idx(k));
    gt = imdsVal.Labels(idx(k));
    if size(I,3)==1, I=cat(3,I,I,I); end
    Iin = center_resize(I, inputSize);
    [pred, ~] = classify(net, Iin);

    nexttile; imshow(I);
    ok = pred==gt;
    ttl = sprintf('GT:%s\nPRED:%s',string(gt),string(pred));
    title(ttl,'Color', ok*[0 0.6 0] + (~ok)*[0.8 0 0], 'FontWeight','bold');
end
end

function J = center_resize(I, outHW)
h=size(I,1); w=size(I,2); s=max(outHW./[h w]);
I2 = imresize(I, s);
y1=floor((size(I2,1)-outHW(1))/2)+1; x1=floor((size(I2,2)-outHW(2))/2)+1;
J = I2(y1:y1+outHW(1)-1, x1:x1+outHW(2)-1, :);
end
