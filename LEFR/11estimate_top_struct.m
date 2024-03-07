
function W = estimate_top_struct(X, K)
% 
% ESTIMATE_TOP_STRUCT      Estimate the topological structure in the feature space.估计特征空间中的拓扑结构。
% 
% Description
%   W = ESTIMATE_TOP_STRUCT(X, K) estimate the topological structure as ML^2. 估计拓扑结构为ML ^ 2。
%   It includes two main steps. First, find K nearest neighbors for each training example. 
%   它包括两个主要步骤。 首先，为每个训练样例找到K个最近邻居。
%   Second, approximate the topological structure of the feature manifold via N standard least square programming problems,
%   其次，通过N个标准最小二乘规划问题逼近特征流形的拓扑结构，其中N是训练样例的数量。
%   where N is the number of training examples. 
% 
% Inputs:
%   X: data matrix with training samples in rows and features in in columns (N x D) 数据矩阵，其中训练样本在行中，并且特征在列中（N×D）
%   K: number of selected nearest neighbors.最近邻居的数量。
%     
% Output:
% 	W: weight matrix 权重矩阵
% 
% Copyright: Peng Hou (hpeng@seu.edu.cn), Xin Geng (xgeng@seu.edu.cn),
%   Min-Ling Zhang (mlzhang@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

fprintf(1,'Estimate the topological structure.\n');%估计拓扑结构。

[N,D] = size(X);

neighborhood = knnsearch(X, X, 'K', K+1);
neighborhood = neighborhood(:, 2:end);

if(K>D) 
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

%---------------------
% M = zeros(N); 
% for i=1:N
%     for j=1:K
%         M(neighborhood(j,i),i)=W(j,i);
%     end;
% end;
% 
% W=M;

[m,N] = size(X);
d=2;K=8;
data=X;

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
end;

end