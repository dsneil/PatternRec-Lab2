%lab 2

clear all % clear all variables from memory
close all % close all open figures

%constants
stepSize = 5;

load('lab2_1.mat');


% instantiating all classes
cA = classData(a, 'r');
cB = classData(b, 'g');

% do a bunch of stuff

clear all;

stepSize = 5;

load('lab2_2.mat');

% instantiating all classes
cA = classData(al, 'r');
cB = classData(bl, 'g');
cC = classData(cl, 'b');

% learn the mean and covariance of the clusters
cA.Mean = Utils.learnMean(cA);
cB.Mean = Utils.learnMean(cB);
cC.Mean = Utils.learnMean(cC);
cA.Var = Utils.learnVariance(cA);
cB.Var = Utils.learnVariance(cB);
cC.Var = Utils.learnVariance(cC);
cA.Cov = Utils.learnCovariance(cA); cA.InvCov = inv(cA.Cov);
cB.Cov = Utils.learnCovariance(cB); cB.InvCov = inv(cB.Cov);
cC.Cov = Utils.learnCovariance(cC); cC.InvCov = inv(cC.Cov);

figure;
Utils.plotClass(cA);
Utils.plotClass(cB);
Utils.plotClass(cC);
hold on;
cont1 = Models2D.gauss2MLEst(stepSize, cA, cB, cC);

figure;
Utils.plotClass(cA);
Utils.plotClass(cB);
Utils.plotClass(cC);
hold on;
[ind, cont, pdfs, xs] = Models2D.parzen2Est(stepSize, cA, cB, cC);

clear all;

stepSize = 5;

load('lab2_3.mat');

cA = classData(a, 'r');
cB = classData(b, 'g');