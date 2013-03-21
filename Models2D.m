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

		function [ind, cont, pdfs, xVs] = parzen2Est(stepSize, windSize, windVar, varargin)
			% Calculates the gaussian parzen window PDF for a given data set
			% --
			% stepSize = resolution of contour for decision boundary
			% windSize = size of gaussian matrix to create [windSize x windSize]
			% windVar = variance of window function
			% varargin = list of classes for evaluation

			[xVals, yVals, ~, ~] = Utils.createGrid(stepSize, varargin{:});
			xmin = min(xVals); xmax = max(xVals); ymin = min(yVals); ymax = max(yVals);
			resVec = [stepSize xmin ymin xmax ymax];

			%fspecial('gaussian', [50 50], sqrt(400));
			gaussWindow = fspecial('gaussian', [windSize windSize], sqrt(windVar));
			% gaussWindow = Utils.gaussian2d(100, 5, 400);

			% Generate each of the pdf's over defined range
			pdfs = []; xVs = []; yVs = [];
			for k=1:length(varargin)
				[p,xVs,yVs] = parzen2(varargin{k}.Cluster, resVec, gaussWindow);
				pdfs = [pdfs p(:)]; % add all points as a column
			end
			[ind, cont] = Utils.parzenMLClassifier('k', xVs, yVs, pdfs);

		end

		function seqContour = seqClassifier(stepSize, cA, cB)
			% Classifies two labelled clusters with sequential linear discriminants.
			% TODO: Return error rates.
			% --
			% stepSize = resolution of contour for decision boundary
			% cA, cB = list of classes for evaluation

			[xVals, yVals, testPts, cont] = Utils.createGrid(stepSize, cA, cB);

			pA = cA.Cluster; pB = cB.Cluster; protoA = []; protoB = [];
			confAs = []; confBs = []; % TODO: error rates
			seqContour = zeros(length(yVals), length(xVals));
			iter = 0;

			while(length(pA) > 0 && length(pB) > 0)
				confA = [0 0]; confB = [0 0]; % H@CK

				while(length(confA) ~= 0 && length(confB) ~= 0)
					protoA = pA(randi(length(pA(:,1)),1),:); 
					protoB = pB(randi(length(pB(:,1)),1),:);

					confA = Utils.LinDiscrimCheck(pA, 1, protoA, protoB);
					confB = Utils.LinDiscrimCheck(pB, 2, protoA, protoB);
				end

				% confAs = [confAs length(confA(:,1))];
				% confBs = [confBs length(confB(:,1))];

				hold on;
				tempCont = Utils.MEDClassifier('--k', xVals, yVals, testPts, cont,...
					true, protoA, protoB);

				mappingVals = [];
				if(length(confA) == 0 && length(confB) == 0),
					mappingVals = tempCont;
					pA = confA; pB = confB;
				elseif(length(confA)==0),
					mod = 2;
					mappingVals = (tempCont == mod).*mod;
					pB = confB;
				elseif(length(confB)==0),
					mod = 1;
					mappingVals = (tempCont == mod).*mod;
					pA = confA;
				else(length(confA) ~= 0 && length(confB) ~= 0), 
					error('failed.'); 
				end

				% - Finds all locations in current contour that are 0
				% - Finds all location in new contour that match our modifier (class)
				% - Multiplies these together to create a composite contour of
				%	where our new contour can "fit" in our current contour
				% - "copies" composite map into our current map.
				seqContour = ((seqContour==0).*(mappingVals)) + seqContour;
			end
			[c, h] = contour(xVals, yVals, seqContour, 3, '--r');
		end
	end
end