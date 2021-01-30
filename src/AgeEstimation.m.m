%% information
% facial age estimation
% regression method: linear regression

%% settings
clear;
clc;

% path 
database_path = './data_age.mat';
result_path = './results/';

% initial states
absTestErr = 0;
cs_number = 0;


% cumulative error level
err_level = 5;

%% Training 
load(database_path);

nTrain = length(trData.label); % number of training samples
nTest  = length(teData.label); % number of testing samples
xtrain = trData.feat; % feature
ytrain = trData.label; % labels

w_lr = regress(ytrain,xtrain);% making model learn the labels 
   
%% Testing
xtest = teData.feat; % feature
ytest = teData.label; % labels

yhat_test = xtest * w_lr; % making model predict them now

%% Compute the MAE and CS value (with cumulative error level of 5) for linear regression 
LRmae = mae(yhat_test,ytest);
LRcum_err = cal_cum_err(yhat_test, ytest);
fprintf('MAE(Linear regression) = %f\n', LRmae);
fprintf('CS(5) = %f\n', LRcum_err(5));
%% generate a cumulative score (CS) vs. error level plot by varying the error level from 1 to 15. The plot should look at the one in the Week6 lecture slides
plot(1:15, LRcum_err(1:15))
%% Compute the MAE and CS value (with cumulative error level of 5) for both partial least square regression and the regression tree model by using the Matlab built in functions.
% Partial least square regression
[XL,yl,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(xtrain, ytrain);
PLSR_yhat_test = [ones(size(xtest,1),1) xtest]*beta;
PLSR_cum_err = cal_cum_err(PLSR_yhat_test, ytest);
PLSR_mae = sum(abs(PLSR_yhat_test-ytest))/size(ytest, 1);
fprintf('MAE(partial least square regression) = %f\n', PLSR_mae);
fprintf('CS(5) = %f\n', PLSR_cum_err(5));

% Regression tree
rt_tree = fitrtree(xtrain, ytrain);
RT_yhat_test = predict(rt_tree, xtest);
RT_cum_err = cal_cum_err(RT_yhat_test, ytest);
RT_mae = sum(abs(RT_yhat_test-ytest))/size(ytest, 1);
fprintf('MAE(regression tree) = %f\n', RT_mae);
fprintf('CS(5) = %f\n', RT_cum_err(5));

%% Compute the MAE and CS value (with cumulative error level of 5) for Support Vector Regression by using LIBSVM toolbox
addpath(genpath('libsvm-3.14'));
svm = svmtrain(ytrain, xtrain, '-s 3 -t 0');
SVM_yhat_test = svmpredict(ytest, xtest, svm);
SVM_cum_err = cal_cum_err(SVM_yhat_test, ytest);
SVM_mae = sum(abs(SVM_yhat_test-ytest))/size(ytest, 1);
fprintf('MAE(SVR) = %f\n', SVM_mae);
fprintf('CS(5) = %f\n', SVM_cum_err(5));

%% Plot
plot(1:15, LRcum_err(1:15), 'g-o'); hold on;
plot(1:15, PLSR_cum_err(1:15) ,'r-o'); hold on;
plot(1:15, RT_cum_err(1:15), 'black-o'); hold on;
plot(1:15, SVM_cum_err(1:15),'b-o'); hold off;
grid on
title('CS plot')
legend('Linear regression','Partial least square regression','Regression tree','SVR')
legend('Location','southeast')
ylabel('Cummulative score')
xlabel('Error levels (year)')