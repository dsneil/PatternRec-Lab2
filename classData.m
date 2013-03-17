classdef classData
    properties
        a
        b
        Mean
        Var
        Cov
        Colour
        Cluster
        InvCov
    end
    methods
        function obj = classData(data, colour)
            obj.Colour = colour;
            obj.Cluster = data;
        end
    end 
end