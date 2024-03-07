%function mod = estimate_top_struct(X, K)
%  clc;
%  clear;
%  load 'set/SJAFFE_binary.mat' ;
%  X=features;
%  K=10;
% T= Tan2_estimate_top_struct(features,K);

function T= estimate_top_struct(X,K)
% ESTIMATE_TOP_STRUCT      Estimate the topological structure in the feature space.估计特征空间中的拓扑结构。
% 
% Description
%   W = ESTIMATE_TOP_STRUCT(X, K) estimate the topological structure.
%   It includes two main steps. First, find K nearest neighbors for each training example. 
%   Second, approximate the topological structure of the feature manifold via N standard least square programming problems,
%   where N is the number of training examples. 
% 
% Inputs:
%   X: data matrix with training samples in rows and features in in columns (N x D) 数据矩阵，其中训练样本在行中，并且特征在列中（N×D）
%   K: number of selected nearest neighbors.最近邻居的数量。
%     
% Output:
% 	W: weight matrix 权重矩阵

fprintf(1,'Estimate the topological structure.\n');%估计拓扑结构。

% X=features;
% K=213;
[N,q] = size(X);

%neighborhood = knnsearch(X, X, 'K', K+1);
neighborhood = knnsearch(X, X, 'K', K);
%neighborhood = neighborhood(:, 2:end);

size(neighborhood);

if(K>q) 
  fprintf(1,'   [note: K>D; regularization will be used]\n'); %正规化将被使用
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned 在情况紧张的情况下，正常情况下病情恶化
else
  tol=0;
end

% Least square programming最小二乘规划
W = sparse(N, N);
for i=1:N
    neighbors = neighborhood(i,:);
    z = X(neighbors,:)-repmat(X(i,:),K,1); % shift ith pt to origin转移到原点
    Z = z*z';                                        % local covariance局部协方差
    Z = Z + eye(K,K)*tol*trace(Z);                   % regularlization (K>D) 正规化（K> D）
    W(i,neighbors) = Z\ones(K,1);                           % solve Zw=1
    W(i,neighbors) = W(i,neighbors)/sum(W(i,neighbors));                  % enforce sum(w)=1
end

% W=W'*W;
% W = sparse(W);

%---------------------LTSA
% M = zeros(N); 
% for i=1:N
%     for j=1:K
%         M(neighborhood(j,i),i)=W(j,i);
%     end;
% end;
% 
% W=M;

% [m,N] = size(X);
% data=X;

 [N,m] = size(X); %********曾改
  data=X';   %********曾改

d=2;K=8;
if nargin<4
    if length(K)==1
       K = repmat(K,[1,N]);
    end;
    NI = cell(1,N);
    if m>N
        a = sum(data.*data); 
        dist2 = sqrt(repmat(a',[1 N]) + repmat(a,[N 1]) - 2*(data'*data));
        for i=1:N
            % Determine ki nearest neighbors of x_j
            [dist_sort,J] = sort(dist2(:,i));  
            Ii = J(1:K(i));
            NI{i} = Ii;
        end;
    else
        for i=1:N
            % Determine ki nearest neighbors of x_j
            x = data(:,i);
            ki = K(i);
            dist2 = sum((data-repmat(x,[1 N])).^2,1);  
            [dist_sort,J] = sort(dist2); 
            Ii = J(1:ki);  
            NI{i} = Ii;
        end;
    end;
else
    K = zeros(1,N);
    for i=1:N
        K(i) = length(NI{i});
    end;
end;

BI = {};
Thera = {};
for i=1:N
    % Compute the d largest right singular eigenvectors of the centered matrix
    Ii = NI{i};
    ki = K(i);
    Xi = data(:,Ii)-repmat(mean(data(:,Ii),2),[1,ki]);
    W = Xi'*Xi;
    W = (W+W')/2;
 %----------------------------------ML222 new   
    [Vi,Si] = schur(W);
    [s,Ji] = sort(-diag(Si)) ;
    Vi = Vi(:,Ji(1:d)) ;
    % construct Gi
    Gi = [repmat(1/sqrt(ki),[ki,1]) Vi] ; 
    % compute the local orthogonal projection Bi = I-Gi*Gi' 
    % that has the null space span([e,Theta_i^T]). 
    BI{i} = eye(ki)-Gi*Gi';   
end;
B = speye(N);
for i=1:N
    Ii = NI{i};
    B(Ii,Ii) = B(Ii,Ii)+BI{i};
    B(i,i) = B(i,i)-1;
end;

W = (B+B')/2;
options.disp = 0; 
options.isreal = 1; 
options.issym = 1; 
[U,D] = eigs(W,d+2,0,options);
lambda = diag(D);
[lambda_s,J] = sort(abs(lambda));
U = U(:,J); lambda = lambda(J);
T = U(:,2:d+1)';

mod.W = W;
mod.T = T;
end