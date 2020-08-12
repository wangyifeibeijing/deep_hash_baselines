clear all;                         ####### matlab command, nothing to do with it

addpath(genpath('minFunc_2012/')); ####### matlab command, nothing to do with it
addpath(genpath('../utils/'));     ####### matlab command, nothing to do with it
dataset = 'cifar10';
bit = 16;                          #set code-length
use_kmeans = 1;                    #use kmeans to fetch anchors
folder_name = '../data_from_ADMM'; ####### matlab command, nothing to do with it
if ~exist(folder_name, 'dir')      ####### matlab command, nothing to do with it
    mkdir(folder_name)
end

load('../fc7_features/traindata.txt'); # matlab load data
%load './cifar_10_gist.mat';
traindata = double(traindata');


X = traindata; % type: single        #normalize X , X is n*dim matrix, n is sample number
X = normalize(X');

traindata = traindata';              #transpose traindata, traindata is dim*n matrix now
sampleMean = mean(traindata, 1);     #mean of each column sampleMean is  dim*1 matrix now 每列的平均值
dim=size(traindata, 1)               #row number of traindata矩阵A的行数
temprep=repmat(sampleMean,dim, 1)    #Replicate Matrix 有B = repmat(A,m,n)，将矩阵 A 复制 m×n 块，即把 A 作为 B 的元素，B 由 m×n 个 A 平铺而成。B 的维数是 [size(A,1)*m, size(A,2)*n] 。
traindata =(traindata - temprep);    #Zero-centered 中心化/零均值化 details in ---- https://www.jianshu.com/p/95a8f035c86c

[n, dim] = size(X);


% parameters  ρ1, ρ2, µ1, µ2, γ.
rho1 = 1e-2;     ρ1
rho2 = 1e-2;     ρ2
rho3 = 1e-3;     µ1
rho4 = 1e-3;     µ2
gamma = 1e-3;    γ
sigma = 0;       ?????
max_iter = 50;   max iteration number
n_anchors = 500; anchor number
s = 2;           % number of nearest anchors, please tune this parameter on different datasets
if ~use_kmeans
     anchor = traindata(randsample(n, n_anchors),:);     ######fetch anchors by sample
else
    fprintf('K-means clustering to get m anchor points\n');
    [~, anchor] = litekmeans(traindata, n_anchors, 'MaxIter', 30);
    fprintf('anchor points have been selected!\n');
    fprintf('---------------------------------------\n');######fetch anchors by kmeans
end

options = [];
options.Display = 'off';
options.MaxFunEvals = 20;
options.Method = 'lbfgs';   % pcg lbfgs   ##L-BFGS算法 有限内存中进行BFGS算法,L是limited memory的意思.details in------ https://blog.csdn.net/weixin_39445556/article/details/84502260

% define ground-truth neighbors
fprintf('Generating anchor graphs\n');
Z = zeros(n, n_anchors);                      #Generating Z , z is n*n_anchors matrix
Dis = sqdist(traindata', anchor');            #Dis n*n_anchors matrix. obtain euclidian distance function D=sqdist(X, Y)   want sth faster, try this: D=bsxfun(@plus,dot(X,X,1)',dot(Y,Y,1))-2*(X'*Y);
%clear X;    
clear traindata;
clear anchor;

val = zeros(n, s);                           #s is the activated neighboors number
pos = val;
for i = 1:s
    [val(:,i), pos(:,i)] = min(Dis, [], 2);  ??????? but I can see we are setting neighboors to very far position except the nearist (s)
    tep = (pos(:,i) - 1) * n + [1:n]';
    Dis(tep) = 1e60;
end
clear Dis;
clear tep;

if sigma == 0
    sigma = mean(val(:,s) .^ 0.5);
end
val = exp(-val / (1 / 1 * sigma ^ 2));
val = repmat(sum(val, 2).^ -1, 1, s) .* val;
tep = (pos - 1) * n + repmat([1:n]', 1, s);
Z([tep]) = [val];
clear tep;
clear val;
clear pos;
% Z = sparse(Z);
lamda = sum(Z);
lambda = diag(lamda .^ -1);
size(lambda)
size(Z)
clear lamda
fprintf('Finished!\n');
fprintf('---------------------------------------\n'); 

%initization
%load('../init_32bits_B/final_32bits.mat');
%B = double(feat_train');
%B = sign(B); 
B = sign(randn(n, bit));  % -1 or 1 random number
init_B = B;
Z1 = B; Z2 = B;
Y1 = rand(n, bit);
Y2 = rand(n, bit);
one_vector = ones(n, 1);
theta1 = zeros(n, bit);
theta2 = zeros(n, bit);
loss_old = 0;
i = 0;

for i = 1:max_iter
%while true
    %i = i + 1;
    fprintf('iteration %3d\n', i);
    % update B
    tic;
    constant = Y1 + Y2 - rho1 * Z1 - rho2 * Z2;
    [Bk_tep, ~, ~, ~] = minFunc(@gradientB, B(:), options, constant, Z, lambda, rho1, rho2, rho3, rho4);
    time = toc;
    count = sum(Bk_tep > 0);
    fprintf('+1, -1: %.2f%%\n', count/n/bit*100);
    fprintf('Update B cost %f seconds\n', time); 
    Bk = reshape(Bk_tep, [n, bit]);
    fprintf('res(init_B and Bk): %d\n', sum(sign(Bk(:))-init_B(:)))

    % update Z1
    tic;
    theta1 = Bk + 1/rho1 * Y1;
    theta1(theta1 > 1) = 1;
    theta1(theta1 < -1) = -1;
%    fprintf('norm theta1(Z1) is %d\n', norm(theta1, 'fro'));
    Z1_k = theta1;

    % update Z2
    theta2 = Bk + 1/rho2 * Y2;
    norm_B = norm(theta2, 'fro');

    theta2 = sqrt(n*bit) * theta2 / norm_B; 
    Z2_k = theta2;
%    fprintf('norm Z2 is %d\n', norm(Z2_k, 'fro'));
    time = toc;
    fprintf('Update Z1 and Z2 cost %f seconds\n', time);  

    %update Y1 and Y2
    tic;
    Y1_k = Y1 + gamma * rho1 * (Bk - Z1_k);
    Y2_k = Y2 + gamma * rho2 * (Bk - Z2_k);
%    fprintf('norm Y2_k is %d\n', norm(Y2_k, 'fro'));
    time = toc;
    fprintf('Update Y1 and Y2 cost %f seconds\n', time);

    B = Bk; 
    Z1 = Z1_k; Z2 = Z2_k; 
    Y1 = Y1_k; Y2 = Y2_k;
      
    res1 = B - Z1; res2 = B - Z2; tmp1 = one_vector'*B; tmp2 = B'*B-n*eye(bit,bit);
    loss = trace(B'*B-B'*Z*lambda*(Z'*B)+Y1'*res1+Y2'*res2)+rho1/2*trace(res1'*res1)...
           +rho2/2*trace(res2'*res2)+rho3/2*trace(tmp1*tmp1')+rho4/4*trace(tmp2'*tmp2);
    res = (loss - loss_old)/loss_old;
    loss_old = loss;
    fprintf('loss is %.4f, residual error is %.5f\n', loss, res);
    fprintf('---------------------------------------\n'); 
    if (abs(res) <= 1e-4)
        break;
    end
end

%clear Y1 Y1_k Y2 Y2_k Z1 Z1_k Z2 Z2_k;

final_B = B;
final_B = sign(final_B);

fprintf('save B and final_B as HDF5 file\n');
fprintf('save path is %s\n' ,save_path);
h5create(save_path, '/final_B',[size(final_B, 2) size(final_B, 1)]);
h5create(save_path, '/B',[size(B, 2) size(B, 1)]);
h5write(save_path, '/final_B', final_B');
h5write(save_path, '/B', B');
fprintf('Finished!\n');                                                         
fprintf('---------------------------------------\n'); 
