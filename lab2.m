%lab 2

% close all % close all open figures
% % clear all % clear all variables from memory


% % %constants
% % stepSize = 5;

% % load('lab2_1.mat');


% % % % instantiating all classes
% % % cA = classData(a, 'r');
% % % cB = classData(b, 'g');

% % % % do a bunch of stuff

% clear all;

% stepSize = 1;

% load('lab2_2.mat');

% % instantiating all classes
% cA = classData(al, 'b');
% cB = classData(bl, 'g');
% cC = classData(cl, 'r');

% % learn the mean and covariance of the clusters
% cA.Mean = Utils.learnMean(cA);
% cB.Mean = Utils.learnMean(cB);
% cC.Mean = Utils.learnMean(cC);
% cA.Cov = Utils.learnCovariance(cA); cA.InvCov = inv(cA.Cov);
% cB.Cov = Utils.learnCovariance(cB); cB.InvCov = inv(cB.Cov);
% cC.Cov = Utils.learnCovariance(cC); cC.InvCov = inv(cC.Cov);

% figure;
% Utils.plotClass(cA);
% Utils.plotClass(cB);
% Utils.plotClass(cC);
% hold on;
% cont1 = Models2D.gauss2MLEst(stepSize, cA, cB, cC);

% legend('Class A', 'Class B', 'Class C', 'Decision Boundary');

% These need to be tweaked
% stepSize = 5;
% windSize = 25;
% windVar = 400;

% figure;
% Utils.plotClass(cA);
% Utils.plotClass(cB);
% Utils.plotClass(cC);
% hold on;
% [ind, cont, pdfs, xs] = Models2D.parzen2Est(stepSize, windSize, windVar, cA, cB, cC);


clear all;

stepSize = 5;

load('lab2_3.mat');

cA = classData(a, 'b');
cB = classData(b, 'g');

figure;
Utils.plotClass(cA);
Utils.plotClass(cB);
hold on;
sC = Models2D.seqClassifier(stepSize, cA, cB);
legend('Class A', 'Class B', 'Discriminants', 'Decision Boundary');

