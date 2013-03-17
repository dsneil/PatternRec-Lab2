classdef classData
    properties
        Mean
        Cov
        Colour
        Cluster
        InvCov
    end
    methods
        function obj = classData(data, colour)
            obj.Mean = [];
            obj.Cov = [];
            obj.InvCov = [];
            obj.Colour = colour;
            obj.Cluster = data;
        end
    end 
end