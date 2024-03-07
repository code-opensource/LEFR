
function W = estimate_top_struct(X, K)
% 
% ESTIMATE_TOP_STRUCT      Estimate the topological structure in the feature space.���������ռ��е����˽ṹ��
% 
% Description
%   W = ESTIMATE_TOP_STRUCT(X, K) estimate the topological structure as ML^2. �������˽ṹΪML ^ 2��
%   It includes two main steps. First, find K nearest neighbors for each training example. 
%   ������������Ҫ���衣 ���ȣ�Ϊÿ��ѵ�������ҵ�K������ھӡ�
%   Second, approximate the topological structure of the feature manifold via N standard least square programming problems,
%   ��Σ�ͨ��N����׼��С���˹滮����ƽ��������ε����˽ṹ������N��ѵ��������������
%   where N is the number of training examples. 
% 
% Inputs:
%   X: data matrix with training samples in rows and features in in columns (N x D) ���ݾ�������ѵ�����������У��������������У�N��D��
%   K: number of selected nearest neighbors.����ھӵ�������
%     
% Output:
% 	W: weight matrix Ȩ�ؾ���
% 
% Copyright: Peng Hou (hpeng@seu.edu.cn), Xin Geng (xgeng@seu.edu.cn),
%   Min-Ling Zhang (mlzhang@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

fprintf(1,'Estimate the topological structure.\n');%�������˽ṹ��

[N,D] = size(X);

neighborhood = knnsearch(X, X, 'K', K+1);
neighborhood = neighborhood(:, 2:end);

if(K>D) 
  fprintf(1,'   [note: K>D; regularization will be used]\n'); %���滯����ʹ��
  tol=1e-3; % regularlizer in case constrained fits are ill conditioned ��������ŵ�����£���������²����
else
  tol=0;
end

% Least square programming��С���˹滮
W = sparse(N, N);
for i=1:N
    neighbors = neighborhood(i,:);
    z = X(neighbors,:)-repmat(X(i,:),K,1); % shift ith pt to originת�Ƶ�ԭ��
    Z = z*z';                                        % local covariance�ֲ�Э����
    Z = Z + eye(K,K)*tol*trace(Z);                   % regularlization (K>D) ���滯��K> D��
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