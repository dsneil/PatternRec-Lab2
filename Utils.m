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
            buffer = 1;
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

            cov = ((1/(length(class.Cluster)*length(data)))*temp);
        end

        function cont = MLClassification(colour, xVals, yVals, testPts, cont, c1, c2, c3) 
            % -Creates a 3 class ML decision boundary
            % -Same implementation as MAP, but priors equal. Could be used for
            %  MAP classification as well
            % --
            % color = colour of decision boundary
            % xVals = range of all x-values to be tested
            % yVals = range of all y-values to be tested
            % testPts = matrix of all points of evaluation. Contains all
            %           combinations of xVals and yVals. Improves performance.
            % cont = Contour map that gets populated with decision boundary.
            %        Will be a matrix of 1's and 2's based on classification.
            % c1, c2, c3 = training data clusters to create boundary.

            function trans = transFunc(pt, mean, invcov)
                trans = (pt - mean)*invcov*(pt - mean)';
            end

            function prob = MLprob(class, pt)
                % prob = transFunc(pt, class.Mean, class.InvCov)+log(det(class.Cov));
                expo = exp((-1/2)*((pt-class.Mean)*(pt-class.Mean)'/class.Var));
                prob = (1/(sqrt(class.Var*2*pi)))*expo;
            end

            xIndex = 1; yIndex = 1; 
            numXs = length(xVals);
            
            for k = 1: length(testPts(:,1))
                pt = testPts(k,:);

                probc1 = MLprob(c1, pt);
                probc2 = MLprob(c2, pt);
                probc3 = MLprob(c3, pt);
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
            
            [c, h] = contour(xVals, yVals, cont', 3, colour);
            %ch = get(h,'child'); alpha(ch,0.5)
        end

        function [ind, cont] = parzenMLClassifier(colour, xVals, yVals, pdfs)
            [~,ind] = max(pdfs, [], 2);

            cont = reshape(ind, length(yVals), length(xVals));

            [c, h] = contour(xVals, yVals, cont, 3, colour);
        end
    end
end