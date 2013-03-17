%Model Estimation 1-D

classdef Utils1D
    methods (Static)

		%Parametric Estimation
		%---------------------

		function [mu, varr] = gaussEst(data, dmu, dvar)
			% Gaussian ML estimation
			% --
			% dmu, dvar = mu, var. set to -1 if unknown
			% data = data set for evaulation
			% 
			% Derivation in notes

			if (dmu == -1)
				mu = ((1/length(data))*sum(data));
			else
				mu = dmu;
			end

			if (dvar == -1)
				temp = 0;
				for k=1:length(data),
					temp = temp + (data(k)-mu)^2;
				end

				varr = ((1/length(data))*temp);
			else
				varr = dvar;
			end
		end

		function lambda = expEst(data)
			% Exponential ML estimation
			% --
			% data = data set for evaluation
			%
			% Derivation in report

			lambda = (1/((1/length(data))*sum(data)));
		end

		function [a, b] = uniformEst(data, da, db)
			% set inputs to -1 if unknown
			% --
			% Want to minimize the range of values, while ensuring all points 
			% are covered in range.

			if da == -1, a = min(data); else, a = da; end;
			if db == -1, b = max(data);	else, b = db; end;	
		end


		%Non-Parametric Estimation
		%-------------------------

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