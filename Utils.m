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
            
            %creates a matrix of all possible vector combinations
            [x,y] = meshgrid(xVals, yVals);
            c = cat(2,x',y');
            testPoints = reshape(c,[],2);
            
            zeroGrid = zeros(length(xVals),length(yVals));
        end

		function trans = transFunc(pt, mean, invcov)
            trans = (pt - mean)*invcov*(pt - mean)';
        end
            
        function pC = probConst(ci, cj)
            tot = ci.N + cj.N;
            pRatio = (cj.N/tot)/(ci.N/tot);
            pC = 2*log(pRatio) + log(det(ci.Cov)/det(cj.Cov));
        end

        function MLClassification(color, xVals, yVals, testPts, cont, c1, c2, c3) 
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

            probConst12 = Utils.probConst(c1, c2);
            probConst23 = Utils.probConst(c2, c3);
            probConst13 = Utils.probConst(c1, c3);
            xIndex = 1; yIndex = 1; 
            numXs = length(xVals);
            
            for k = 1: length(testPts(:,1))
                pt = testPts(k,:);
                transC1 = Utils.transFunc(pt, c1.Mean, c1.InvCov);
                transC2 = Utils.transFunc(pt, c2.Mean, c2.InvCov);
                transC3 = Utils.transFunc(pt, c3.Mean, c3.InvCov);

                if (((transC2 - transC1)>(probConst12)) && ...
                        ((transC3 - transC1)>(probConst13)))
                    cont(xIndex,yIndex) = 1;
                elseif (((transC2 - transC1)<(probConst12)) && ...
                        ((transC3 - transC2)>(probConst23)))
                    cont(xIndex,yIndex) = 2;
                else
                    cont(xIndex,yIndex) = 3;
                end
                
                if(xIndex == numXs)
                    xIndex = 1;
                    yIndex = yIndex +1;
                else
                    xIndex = xIndex +1;
                end
            end
            
            [c, h] = contour(xVals, yVals, cont', 3, color);
            %ch = get(h,'child'); alpha(ch,0.5)
        end
    end
end