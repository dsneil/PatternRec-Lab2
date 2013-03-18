%Model Estimation 2-D

classdef Models2D
    methods (Static)

    	
		% Parametric Estimation
		% ---------------------

		function cont1 = gauss2MLEst(stepSize, cA, cB, cC)
			% Gaussian ML estimation
			% ->Currently only functional for 3 classes (ML needs abstraction)
			% --
			% stepSize = resolution of contour for decision boundary
			% varargin = list of classes for evaluation

			[xVals, yVals, testPts, cont] = Utils.createGrid(stepSize, cA, cB, cC);

			cont1 = Utils.MLClassification('k',xVals,yVals,testPts,cont,cA,cB,cC);
		end

		% Non-Parametric Estimation
		% -------------------------

		function [ind, cont, pdfs, xVs] = parzen2Est(stepSize, varargin)
			% Calculates the gaussian parzen window PDF for a given data set
			% --
			% data = data set
			% sigma = standard deviation of parzen window
			% res = resolution (step size of evaluation)
			% buff = buffer on either side of data set. used in creating
			%	     evaluation set

			[xVals, yVals, ~, ~] = Utils.createGrid(stepSize, varargin{:});
			xmin = min(xVals); xmax = max(xVals); ymin = min(yVals); ymax = max(yVals);
			resVec = [stepSize xmin ymin xmax ymax];

			% creates a 50x50 gaussian window with variance 400 fspecial('gaussian', [50 50], sqrt(400));
			gaussWindow = Utils.gaussian2d(100, 5, 400);

			% Generate each of the pdf's over defined range
			pdfs = []; xVs = []; yVs = [];
			for k=1:length(varargin)
				[p,xVs,yVs] = parzen2(varargin{k}.Cluster, resVec, gaussWindow);
				pdfs = [pdfs p(:)];
			end

			[ind, cont] = Utils.parzenMLClassifier('k', xVs, yVs, pdfs);

		end

		function sequentialClassifier(stepSize, varargin)

		end
	end
end