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
			% 	 evaluation set

			[xVals, yVals, ~, ~] = Utils.createGrid(stepSize, varargin{:});
			xmin = min(xVals); xmax = max(xVals); ymin = min(yVals); ymax = max(yVals);
			resVec = [stepSize xmin ymin xmax ymax];

			% creates a 50x50 gaussian window with variance 400 
			%fspecial('gaussian', [50 50], sqrt(400));
			gaussWindow = Utils.gaussian2d(100, 5, 400);

			% Generate each of the pdf's over defined range
			pdfs = []; xVs = []; yVs = [];
			for k=1:length(varargin)
				[p,xVs,yVs] = parzen2(varargin{k}.Cluster, resVec, gaussWindow);
				pdfs = [pdfs p(:)]; % add all points as a column
			end

			[ind, cont] = Utils.parzenMLClassifier('k', xVs, yVs, pdfs);

		end

		function seqContour = seqClassifier(stepSize, cA, cB)
			[xVals, yVals, testPts, cont] = Utils.createGrid(stepSize, cA, cB);

			pA = cA.Cluster; pB = cB.Cluster;
			protoA = []; protoB = [];
			seqContour = zeros(length(xVals), length(yVals));
			iter = 0;

			while(length(pA) > 0 && length(pB) > 0)
				confA = [0 0]; confB = [0 0]; % H@CK

				while(length(confA) ~= 0 && length(confB) ~= 0)
					protoA = pA(randi(length(pA(:,1)),1),:); 
					protoB = pB(randi(length(pB(:,1)),1),:);

					confA = Utils.LinDiscrimCheck(pA, 1, protoA, protoB);
					confB = Utils.LinDiscrimCheck(pB, 2, protoA, protoB);
				end

				if(length(confA)==0), pB = confB; end
				if(length(confB)==0), pA = confA; end
				if(length(confA) ~= 0 && length(confB) ~= 0), error('fails'); end;

				hold on;
				tempCont = Utils.MEDClassifier('k', xVals, yVals, testPts, cont,...
					protoA, protoB);

				[vs,ind] = max([tempCont(:) seqContour(:)], [], 2);

            	seqContour = reshape(vs, length(yVals), length(xVals));

			end
			[c, h] = contour(xVals, yVals, seqContour, 3, 'b');
		end
	end
end