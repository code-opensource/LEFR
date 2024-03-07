%PLEADEMO	
clear;
clc;

load('finaldata.mat'); %-----------yeast
trainX=Xapp;
trainY=Yapp;
testX=Xgen;
testY=Ygen;

% load('image.mat'); 

% load('emotions.mat'); 
% trainX=train_data;
% trainY=train_targets;
% testX=test_data;
% testY=test_targets;
% trainY(trainY==0)=-1;
% testY(testY==0)=-1;

% load('scene.mat');
% trainX=features(1:1211,:);
% trainY=labels(1:1211,:);
% testX=features(1212:2407,:);
% testY=labels(1212:2407,:);
% trainY(trainY==0)=-1;
% testY(testY==0)=-1;

% load('medical.mat');
% trainX=features(1:645,:);
% trainY=labels(1:645,:);
% testX=features(646:978,:);
% testY=labels(646:978,:);
% trainY(trainY==0)=-1;
% testY(testY==0)=-1;

% load('enron.mat');
% trainX=features(1:1123,:);
% trainY=labels(1:1123,:);
% testX=features(1124:1702,:);
% testY=labels(1124:1702,:);
% trainY(trainY==0)=-1;
% testY(testY==0)=-1;

% load('cal500.mat');
% trainX=features(1:250,:);
% trainY=labels(1:250,:);
% testX=features(251:502,:);
% testY=labels(251:502,:);
% trainY(trainY==0)=-1;
% testY(testY==0)=-1;

% load('corel5k.mat');
% trainX=features(1:2500,:);
% trainY=labels(1:2500,:);
% testX=features(2501:5000,:);
% testY=labels(2501:5000,:);
% trainY(trainY==0)=-1;
% testY(testY==0)=-1;
 
% load('bibtex.mat');
% trainX=features(1:3700,:);
% trainY=labels(1:3700,:);
% testX=features(3701:7395,:);
% testY=labels(3701:7395,:);
% trainY(trainY==0)=-1;
% testY(testY==0)=-1;

% load('birds.mat');
% trainX=features(1:320,:);
% trainY=labels(1:320,:);
% testX=features(321:645,:);
% testY=labels(321:645,:);
% trainY(trainY==0)=-1;
% testY(testY==0)=-1;

% load movieDataSet;

% Load the data set.
% load movieDataSet;

% load('Yeast_alpha.mat');
% trainFeature=features(1:1232,:);
% trainDistribution=labels(1:1232,:);
% testFeature=features(1233:2465,:);
% testDistribution=labels(1233:2465,:);

% load('Human_Gene.mat');
% trainFeature=features(1:8946,:);
% trainDistribution=labels(1:8946,:);
% testFeature=features(8947:17892,:);
% testDistribution=labels(8947:17892,:);

% load('Natural_Scene.mat');
% trainFeature=features(1:1000,:);
% trainDistribution=labels(1:1000,:);
% testFeature=features(1001:2000,:);
% testDistribution=labels(1001:2000,:);

% load('Movie.mat');
% trainFeature=features(1:3877,:);
% trainDistribution=labels(1:3877,:);
% testFeature=features(3878:7755,:);
% testDistribution=labels(3878:7755,:);

% load('SJAFFE.mat');
% trainFeature=features(1:106,:);
% trainDistribution=labels(1:106,:);
% testFeature=features(106:213,:);
% testDistribution=labels(106:213,:);

% load('SBU_3DFE.mat');
% trainFeature=features(1:1250,:);
% trainDistribution=labels(1:1250,:);
% testFeature=features(1250:2500,:);
% testDistribution=labels(1250:2500,:);


% load('Berkeley1_n.mat');
% trainFeature=train_data;
% trainDistribution=train_target';
% testFeature=test_data;
% testDistribution=test_target';

% trainX=trainFeature;
% trainY=trainDistribution;
% testX=testFeature;
% testY=testDistribution;
% trainY(trainY==0)=-1;
% testY(testY==0)=-1;

train_data=trainX;
train_target=trainY;
test_data=testX;
test_target=testY;

trainFeature=trainX;
trainDistribution=trainY;
testFeature=testX;
testDistribution=testY;

% Initialize the model parameters.
para.minValue = 1e-7; % the feature value to replace 0, default: 1e-7
para.iter = 10; % learning iterations, default: 50 / 200 
para.minDiff = 1e-4; % minimum log-likelihood difference for convergence, default: 1e-7
para.regfactor = 0; % regularization factor, default: 0

tic;
% The training part of IISLLD algorithm.

% initialize the model parameters.

para.tol  = 1e-5; %tolerance during the iteration 在迭代期间容差
para.epsi = 0.001; %instances whose distance computed is more than epsi should be penalized 其计算的距离大于epsi的实例应该受到惩罚
para.C1    = 1; %penalty parameter惩罚参数
para.C2    = 10; %penalty parameter惩罚参数
para.ker  = 'rbf'; %type of kernel function ('lin', 'poly', 'rbf', 'sam')核函数的类型
para.par  = 1*mean(pdist(abs(trainX))); %parameter of kernel function核函数的参数
%K = size(trainX,1);
K=8;
lambda = 1;

W = estimate_top_struct(abs(trainX), K);   %估计特征空间的拓扑结构，输出权重矩阵。
MU = build_label_manifold(trainY, W, lambda);  %转移拓扑结构到标签空间并构造标签流形，输出数字标签矩阵（N×L）。
%mod.MU=trainDistribution;
size(MU);
%[weights] = iislldTrain(para, trainFeature, MU);%PLEA-
%[weights] = iislldTrain(para, T, MU);%PLEA
model = amsvr(trainX, MU, para);
[label, degree,te_time] = predict(testX, trainX, model);

fprintf('Training time of IIS-LLD: %8.7f \n', toc);
%preDistribution = lldPredict(weights,testFeature);
fprintf('Finish prediction of IIS-LLD. \n');

% Prediction
% preDistribution = lldPredict(weights,testFeature);
% fprintf('Finish prediction of PLEA. \n');

% To visualize two distribution and display some selected metrics of distance
% testNum=5;
% for i=1:testNum
%     % Show the comparisons between the predicted distribution
% 	[disName, distance] = computeMeasures(testDistribution(i,:), preDistribution(i,:));
%     % Draw the picture of the real and prediced distribution.
%     drawDistribution(testDistribution(i,:),preDistribution(i,:),disName, distance);
%     %sign=input('Press any key to continue:');
% end
Outputs=degree;
size(Outputs);
Pre_Labels=label;

HammingLoss=Hamming_loss(Pre_Labels,test_target);   
RankingLoss=Ranking_loss(Outputs,test_target);
OneError=One_error(Outputs,test_target);
Coverage=coverage(Outputs,test_target);
Average_Precision=Average_precision(Outputs,test_target);
