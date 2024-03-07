function MU = build_label_manifold(Y, W, lambda)
% 
% BUILD_LABEL_MANIFOLD      The label manifold building part of the algorithm ML^2.算法ML2的标签流形构建部分。
% 
% Description
%   MU = BUILD_LABEL_MANIFOLD(X, Y, K, lambda) is the label manifold building part of the algorithm ML^2. 是算法的标签流形构建部分。
%   It constructs the label manifold via L quadratic programming problems.它通过L二次规划问题构造标签流形。
% 
% Inputs:
%   Y: multi-label matrix corresponding to the training samples in X above (N x L). Note that each element 
%   in this matrix can only take 1 or -1, where 1 represents the corresponding label is relevant and -1 represents 
%   the corresponding label is irrelevant.请注意，每个元素在这个矩阵中只能取1或-1，其中1代表相应的标签是相关的，-1代表相应的标签是不相关的。
%   W: weight matrix
%	lambda: parameter in the constraint (3) in our paper.约束（3）参数在我们的论文。
%     
% Output:
% 	MU: constructed numerical labels.构建数字标签。
% 
% Copyright: Peng Hou (hpeng@seu.edu.cn), Xin Geng (xgeng@seu.edu.cn),
%   Min-Ling Zhang (mlzhang@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%

fprintf(1,'Build the label manifold.\n');

[N, L] = size(Y);
M=speye([N,N]);%K=8;
%-------------------Houpeng
% for i=1:N
%    w = W(i,:);
%    M(i,:) = M(i,:) - w;
%    M(:,i) = M(:,i) - w';
%    M = M + w'*w;
% end
% 
% M=M'*M;
% M = sparse(M);
% %---------------------CC
% M = zeros(N); 
% for i=1:N
%     for j=1:K
%         M(neighborhood(j,i),i)=W(j,i);
%     end;
% end;

%CC:
M=M';
R=M+M'-M'*M;
%L = eye(N)-R;
M = R-eye(N);

%   [m,n]  = size(M);
%   %[m1,n1] = size(T);
%   normal = zeros(m,n);
%   %normal1 = zeros(m1,n1);
%   for i = 1:m
%         mea = mean( M(i,:) );
%         va = var( M(i,:) );
%         normal(i,:) = ( M(i,:)-mea )/va;
%   end

  H=eye(N)-(1/N)*ones(N,1)*ones(1,N);
  T=-H*M*H/2;
  
  M=1/2*(L+T);
  %M=(1.0e+0)*L-1*(1.0e-6)*T;
  
  M=M';
  R=M+M'-M'*M;

  M = eye(N)-R;

 
% For sparse datasets, we might end up with NaNs or Infs in M. We just set them to zero for now...
%对于稀疏的数据集，我们可能会以M的NaN或Infs结尾。现在我们将它们设置为零。
M(isnan(M)) = 0;
M(isinf(M)) = 0;

% Quadratic programming
b = zeros(N,1)-lambda;
options = optimoptions('quadprog',...
    'Display', 'off');
for k=1:L
    A = -diag(Y(:,k));
    MU(:,k) = quadprog(2*M, [], abs(A), b, [], [], [], [],[], options);
    %MU(:,k) = quadprog(2*M, [], A, b, [], [], [], [],[], options);
end

end