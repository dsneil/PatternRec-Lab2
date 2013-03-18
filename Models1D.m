%Model Estimation 1-D

classdef Models1D
    methods (Static)

		% Parametric Estimation
		% ---------------------

		function [mu, varr] = gaussEst(class)
			% Gaussian ML estimation
			% --
			% class = classData used for estimating mean and var

			class.Mean = Utils.learnMean(class);
			class.Var = Utils.learnVariance(class);
		end

		function lambda = expEst(class)
			% Exponential ML estimation
			% --
			% class = data set for evaluation
			%
			% Derivation in report

			lambda = (1/((1/length(class.Cluster))*sum(class.Cluster)));
		end

		function [a, b] = uniformEst(class)
			% Want to minimize the range of values, while ensuring all points 
			% are covered in range.

			if isempty(class.a) == 1, a = min(class.Cluster); 
			else, a = class.a; 
			end;

			if isempty(class.b) == 1, class.b = max(class.Cluster); 
			else, b = class.b; 
			end;	
		end


		% Non-Parametric Estimation
		% -------------------------

		function [x, pdf] = parzen1Est(data, sigma, res, buff)
			% Calculates the gaussian parzen window PDF for a given data set
			% --
			% data = classData cluster data. data set of given class
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
					% try using the dot operator after
				end
				wind =[wind temp];
			end	
			pdf = (1/(N*sigma*sqrt(2*pi)))*wind;
		end
	end
end