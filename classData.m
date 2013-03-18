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
            a = 0;
            b = 0;
            Mean = [];
            Var = [];
            Cov = [];
            obj.Colour = colour;
            obj.Cluster = data;
        end
    end 
end