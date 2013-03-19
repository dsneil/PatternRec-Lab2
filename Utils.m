classdef Utils
    methods (Static)

        function plotClass(Class)
            % Simplifies plotting process. Plots on existing figure.

            hold on;
            scatter(Class.Cluster(:,1), Class.Cluster(:,2), 5, ...
                Class.Colour, 'filled');
        end

		function [xVals, yVals, testPoints, zeroGrid] = createGrid(stepSize, varargin)
            % -Creates a grid of zeros to be used in classification procedures
            % -Returns a list of xVals and yVals containing all points of eval
            % -All combinations are contained in testPoints
            % -ZeroGrid is the contour to be populated in classification
            % --
            % stepSize = resolution of zeroGrid
            % varargin = list of classes used to create grid

            mixedVals = [];
            buffer = 5;
            for k = 1 : length(varargin)
                %appends all cluster data into mixedVals matrix
                mixedVals = [mixedVals; varargin{k}.Cluster];
            end
            
            %extracts min/max x and y from mixed data
            mins = min(mixedVals);
            maxs = max(mixedVals);
            
            xVals = mins(:,1)-buffer:stepSize:maxs(:,1)+buffer;
            yVals = mins(:,2)-buffer:stepSize:maxs(:,2)+buffer;
            
            [x,y] = meshgrid(xVals, yVals);
            c = cat(2,x',y');
            testPoints = reshape(c,[],2);
            
            zeroGrid = zeros(length(xVals),length(yVals));
        end

        function testPts = createEvalMtx(xVals, yVals)
            %creates a matrix of all possible vector combinations
            
            [x,y] = meshgrid(xVals, yVals);
            c = cat(2,x',y');
            testPts = reshape(c,[],2);
        end

        function func = gaussian2d(N, stepSize, var)
            % creates a 2D gaussian window function
            % --
            % N = size of matrix
            % stepSize = resolution of window function
            % var = variance of desired gaussian

            [x y] = meshgrid(round(-N/2):stepSize:round(N/2));

            func = exp(-x.^2/(2*var)-y.^2/(2*var));
            func = func./sum(func(:)); % normalizes the function
        end

        function prob = gauss2D(class, pt)
            % prob = transFunc(pt, class.Mean, class.InvCov)+log(det(class.Cov));
            % (pt-class.Mean)*(pt-class.Mean)'/det(class.Cov))
            % may need to change to covariance
            trans = (pt - class.Mean)*class.InvCov*(pt - class.Mean)';
            expo = exp((-1/2)*(trans));
            prob = (1/(det(class.Cov)*sqrt(2*pi)))*expo;
        end

        function mu = learnMean(class)
            % Learns the mean of a data set
            mu = ((1/length(class.Cluster))*sum(class.Cluster));
        end

        function var = learnVariance(class)
            % Learns the variance of a data set
            if (isempty(class.Var) == 1)
                temp = 0;
                data = class.Cluster;
                for k=1:length(data),
                    temp = temp + (data(k,:)-class.Mean)*(data(k,:)-class.Mean)';
                end

                var = ((1/(length(class.Cluster)*length(data)))*temp);
            else
                var = dvar;
            end
        end

        function [cov, data] = learnCovariance(class)
            % Learns the covariance matrix of a data set
            temp = [0 0; 0 0]; %set defaults
            data = class.Cluster;

            for k=1:length(data),
                temp = temp + (data(k,:)-class.Mean)'*(data(k,:)-class.Mean);
            end

            cov = ((1/(length(data)))*temp);
        end

        function eD = eucD(p1, p2)
            eD = sqrt(sum((p1 - p2) .^ 2));
        end

        %--------------------
        % CLASSIFIERS
        %--------------------

        function cont = MLClassification(colour, xVals, yVals, testPts, cont, c1, c2, c3) 
            % Creates a 3 class ML decision boundary
            % --
            % color = colour of decision boundary
            % xVals = range of all x-values to be tested
            % yVals = range of all y-values to be tested
            % testPts = matrix of all points of evaluation. Contains all
            %           combinations of xVals and yVals. Improves performance.
            % cont = Contour map that gets populated with decision boundary.
            %        Will be a matrix based on classification of points.
            % c1, c2, c3 = training data clusters to create boundary.

            xIndex = 1; yIndex = 1;
            numXs = length(xVals);
            
            for k = 1: length(testPts(:,1))
                pt = testPts(k,:);

                probc1 = Utils.gauss2D(c1, pt);
                probc2 = Utils.gauss2D(c2, pt);
                probc3 = Utils.gauss2D(c3, pt);
                vals = [probc1 probc2 probc3];
                [~, ind] = max(vals);

                cont(xIndex, yIndex) = ind;
                
                if(xIndex == numXs)
                    xIndex = 1;
                    yIndex = yIndex +1;
                else
                    xIndex = xIndex +1;
                end
            end
            
            [c, h] = contour(xVals, yVals, cont', 2, colour);
            % ch = get(h,'child'); alpha(ch,0.5)
        end

        function [ind, cont] = parzenMLClassifier(colour, xVals, yVals, pdfs)
            % Classifies points within a contour, given a set of pdfs that have
            % been processed using a 2D parzen window. (can work with other pdfs)
            % --
            % colour = colour of decision boundary
            % xVals = range of all x-values tested
            % yVals = range of all y-values tested
            % pdfs = set of class pdfs to be used in classification. Must be
            %        inputted as column of all points (e.g. pdf1(:)).

            [~,ind] = max(pdfs, [], 2);

            cont = reshape(ind, length(yVals), length(xVals));

            [c, h] = contour(xVals, yVals, cont, 3, colour);
        end

        function finCont = MEDClassifier(colour, xVals, yVals, testPts, cont, plotFlag, varargin)
            % Minimum Euclidian Distance classifier.
            % --
            % colour = colour of decision boundary
            % xVals = range of all x-values to be tested
            % yVals = range of all y-values to be tested
            % testPts = matrix of all points of evaluation. Contains all
            %           combinations of xVals and yVals. Improves performance.
            % cont = Contour map that gets populated with decision boundary.
            %        Will be a matrix based on classification of points.
            % varargin = set of class prototypes.

            dists = [];
            xIndex = 1; yIndex = 1; numXs = length(xVals);
            for k = 1: length(testPts(:,1))
                for s = 1 : length(varargin)
                    dists = [ dists Utils.eucD(testPts(k,:),varargin{s})];
                end
                [~, minClass] = min(dists);
                cont(xIndex,yIndex) = minClass;
                dists = [];
                
               if(xIndex == numXs)
                    xIndex = 1;
                    yIndex = yIndex +1;
                else
                    xIndex = xIndex +1;
                end
            end 
            
            finCont = cont';
            if(plotFlag)
                [c, h] = contour(xVals,yVals, cont', 2, colour);
            end
            %ch = get(h,'child'); alpha(ch,0.05);
            
        end

        function confPoints = LinDiscrimCheck(pCl, index, varargin)
            % Determines the incorrectly classified points from a linear
            % discriminant function. Essentially confusion matrix entries
            % of incorrectly classified points.
            % --
            % pCl = points in the class to be analysed
            % index = index of class in varargin that is being analysed
            % varargin = set of class prototypes used in analysis
            %
            % e.g. LinDiscrimCheck(pointsA, 1, protoA, protoB)
            %   --> Will return a set of points in A that are classified as B
            %       with current prototype values

            confPoints = []; dists = [];

            for k=1:length(pCl(:,1))
                for s = 1 : length(varargin)
                    dists = [ dists Utils.eucD(pCl(k,:),varargin{s})];
                end
                [~, minClass] = min(dists);
                dists = [];

                % Add incorrectly classified points to set of confusion points
                if(minClass ~= index)
                    confPoints = [confPoints; pCl(k,:)];
                end
            end
        end
    end
end