function MU = build_label_manifold(Y, W, lambda)
% 
% BUILD_LABEL_MANIFOLD      The label manifold building part of the algorithm ML^2.�㷨ML2�ı�ǩ���ι������֡�
% 
% Description
%   MU = BUILD_LABEL_MANIFOLD(X, Y, K, lambda) is the label manifold building part of the algorithm ML^2. ���㷨�ı�ǩ���ι������֡�
%   It constructs the label manifold via L quadratic programming problems.��ͨ��L���ι滮���⹹���ǩ���Ρ�
% 
% Inputs:
%   Y: multi-label matrix corresponding to the training samples in X above (N x L). Note that each element 
%   in this matrix can only take 1 or -1, where 1 represents the corresponding label is relevant and -1 represents 
%   the corresponding label is irrelevant.��ע�⣬ÿ��Ԫ�������������ֻ��ȡ1��-1������1������Ӧ�ı�ǩ����صģ�-1������Ӧ�ı�ǩ�ǲ���صġ�
%   W: weight matrix
%	lambda: parameter in the constraint (3) in our paper.Լ����3�����������ǵ����ġ�
%     
% Output:
% 	MU: constructed numerical labels.�������ֱ�ǩ��
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
%����ϡ������ݼ������ǿ��ܻ���M��NaN��Infs��β���������ǽ���������Ϊ�㡣
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