function J = GMMobjective(Params, Model, W)
% =============================================================================================
% Objective function for CKLS nested models
% 
% INPUT: Params, vector, vector of estimated parameters
%        Model, structure, see RunAssignment2  
%        W, matrix, weighting matrix
% OUTPUT: d, Jacobian matrix
% =============================================================================================

Data = Model.Data;
DataF = Data(2:end);
DataL = Data(1:end-1);
Nobs = length(DataL);
Nobs = Nobs-1;
TimeStep = Model.TimeStep;
a = Params(1);
b = Params(2);

% Calculate the sample moments 
switch Model.Name
    case 'CKLS' 
        sigma = Params(3);
        gamma = Params(4);
        g1 = sum(DataF - a - b*DataL);
        g2 = sum((DataF - a - b*DataL).^2 - sigma^2*DataL.^(2*gamma)*TimeStep);
        g3 = sum((DataF - a -b*DataL).*DataL);
        g4 = sum(((DataF - a - b*DataL).^2 - sigma^2*DataL.^(2*gamma)*TimeStep).*DataL);        
        g1 = g1/Nobs; g2 = g2/Nobs; g3 = g3/Nobs; g4 = g4/Nobs;  
    case 'CIR'
        sigma = Params(3);
        g1 = sum(DataF - a - b*DataL);
        g2 = sum((DataF - a - b*DataL).^2 - sigma^2*DataL.*TimeStep);
        g3 = sum((DataF - a -b*DataL).*DataL);
        g4 = sum(((DataF - a - b*DataL).^2 - sigma^2*DataL.*TimeStep).*DataL);        
        g1 = g1/Nobs; g2 = g2/Nobs; g3 = g3/Nobs; g4 = g4/Nobs;  
    case 'Vasicek'
        sigma = Params(3);
        g1 = sum(DataF - a - b*DataL);
        g2 = sum((DataF - a - b*DataL).^2 - sigma^2*TimeStep);
        g3 = sum((DataF - a -b*DataL).*DataL);
        g4 = sum(((DataF - a - b*DataL).^2 - sigma^2*TimeStep).*DataL);        
        g1 = g1/Nobs; g2 = g2/Nobs; g3 = g3/Nobs; g4 = g4/Nobs;         
end
g = [g1 g2 g3 g4];                                       
J = g*W*g';

end

