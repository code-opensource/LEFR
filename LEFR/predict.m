function [label, degree,te_time] = predict(testX, trainX, model)
%PREDICT	Calculate the predicted numerial label values for testX.
%
%	Description
%   PREDICTION = PREDICT(TESTX, TRAINX, MODEL) calculate 
%   the predicted numerial label values for testX.
%
%   Inputs,
%       TESTX:  data matrix with test samples in rows and features in in columns (N1 x D)
%       TRAINX: data matrix with training samples in rows and features in columns (N2 x D)
%       MODEL:  model parameters of AMSVR model.
%
%   Outputs,
%       LABEL:  predicted labels for testX.
%       DEGREE: predicted label degrees for testX. Note that the above
%       LABEL is determined by the sign of DEGREE.
%
%   Extended description of input/ouput variables
%   MODELPARA,
%       MODEL.SVINDEX : support vectors' subscripts of row in trainX
%       MODEL.BETA :    coeficient matrix of trainX's linear combination (N2 x L)
%       MODEL.B :       intercept matrix (1 x L)
%       MODEL.KER :     type of kernel function ('lin', 'poly', 'rbf', 'sam')
%       MODEL.PAR :     parameters of kernel function
%           SIGMA:  width of the RBF and sam kernel
%           BIAS:   bias in the linear and polinomial kernel
%           DEGREE: degree in the polynomial kernel
%
%
%   Copyright: Peng Hou (hpeng@seu.edu.cn), Xin Geng (xgeng@seu.edu.cn),
%   Min-Ling Zhang (mlzhang@seu.edu.cn)
%   School of Computer Science and Engineering, Southeast University
%   Nanjing 211189, P.R.China
%
start_time=cputime;

fprintf(1,'\nPredict multi-label for the test data.\n');

%Compute kernel matrix for prediction using testX and trainX
Ktest = kernelmatrix(model.ker, model.par, testX, trainX);
%Prediction.
% size(trainX)
% size(model.Beta)
% size(model.b)
% size(Ktest,1)
% degree = trainX'*model.Beta+repmat(model.b,size(Ktest,1),1);
degree = Ktest*model.Beta+repmat(model.b,size(Ktest,1),1);

label = zeros(size(degree));
label(degree >= 0) = 1;
label(degree < 0) = -1;
te_time=cputime-start_time;
end