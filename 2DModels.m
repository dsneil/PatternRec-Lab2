%Model Estimation 2-D

classdef Utils2D
    methods (Static)

		% Parametric Estimation
		% ---------------------

		function gauss2MLEst(stepSize, varagin)
			% Gaussian ML estimation
			% ->Currently only functional for 3 classes (ML needs abstraction)
			% --
			% stepSize = resolution of contour for decision boundary
			% varargin = list of classes for evaluation

			cs = varargin

			% learn the mean and covariance of the clusters
			for c=1:length(cs)
				[cs(c).Mean, cs(c).Cov] = estMuCov(cs(c));
				cs(c).InvCov = inv(cs(c).Cov)
			end

			[xVals, yVals, testPts, cont] = Utils.createGrid(stepSize, varargin);

			Utils.MLClassification( ...
					colour, ...
					xVals, ...
					yVals, ...
					testPts, ...
					cont, ...
					varargin(1), ...
					varargin(2), ...
					varargin(3), ...
				);
		end

		function [mu, cov] = estMuCov(class)
			if (class.Mean == 0)
				mu = ((1/length(data))*sum(data));
			else
				mu = class.Mean;

			if (class.Cov == 0)
				temp = 0;
				for k=1:length(data),
					temp = temp + (data(k)-mu)*(data(k)-mu)';
				end

				cov = ((1/length(data))*temp);
			else
				cov = class.Cov;
		end

		% Non-Parametric Estimation
		% -------------------------

		function [x, pdf] = parzen1(data, sigma, res, buff)
			% Calculates the gaussian parzen window PDF for a given data set
			% --
			% data = data set
			% sigma = standard deviation of parzen window
			% res = resolution (step size of evaluation)
			% buff = buffer on either side of data set. used in creating
			%	     evaluation set

			N = length(data);
			x = [min(data)-buff:res:max(data)+buff];

			wind = [];
			for k=1:length(x),
				temp = 0;
				for v=1:length(data),
					temp = temp + exp((-1/2)*(((x(k)-data(v))'*(x(k)-data(v)))/sigma^2));
				end
				wind =[wind temp];
			end	
			pdf = (1/(N*sigma*sqrt(2*pi)))*wind;
		end
	end
end