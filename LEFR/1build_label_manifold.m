function MU = build_label_manifold(Y, W, lambda)
% 
% BUILD_LABEL_MANIFOLD      The label manifold building part.
% 
% Description
%   MU = BUILD_LABEL_MANIFOLD(X, Y, K, lambda) is the label manifold building part. ���㷨�ı�ǩ���ι������֡�
%  
% Inputs:
%   Y: multi-label matrix corresponding to the training samples in X above (N x L). 
%   W: weight matrix
%	lambda: parameter in the constraint
%     
% Output:
% 	MU: constructed numerical labels.

fprintf(1,'Build the label manifold.\n');

[N, L] = size(Y);
M=speye([N,N]);%K=8;

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
% --------------------new
  H=eye(N)-(1/N)*ones(N,1)*ones(1,N);
  T=-H*M*H/2;
  
  M=1/2*(L+T);
  %M=(1.0e+0)*L-1*(1.0e-6)*T;
  
  M=M';
  R=M+M'-M'*M;

  M = eye(N)-R;

 
% For sparse datasets, we might end up with NaNs or Infs in M. We just set them to zero for now...

M(isnan(M)) = 0;
M(isinf(M)) = 0;

% Quadratic programming
b = zeros(N,1)-lambda;
options = optimoptions('quadprog',...
    'Display', 'off');
for k=1:L
    %A = -diag(Y(:,k));
    %MU(:,k) = quadprog(2*M, [], A, b, [], [], [], [],[], options);
    A = -diag(Y(:,k)); %size(Y(:,k))=N,1 %size(A)=N,N
    MU(:,k) = quadprog(2*M, [], A, b, [], [], [], [],[], options);%size(MU(:,k))=320,1
    %MU(:,k)=abs(MU(:,k));
end

end