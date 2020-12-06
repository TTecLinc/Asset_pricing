function Results = GMMestimation(Model) 
% =============================================================================================
% GMM estimation routine for CKLS nested models
% 
% INPUT: Model,structure, see RunAssignment2
% OUTPUT: Results, structure 
%         Results.Params, estimated parameters
%         Results.Fval, objective function value
%         Results.Exitflag, Matlab's optimization result  
%         Results.Tstat, t-statistics for individual parameters
%         Results.Chi2statisitcs, Chi2 test of model specificaion
% USES: GMMobjective, GMMweightsNW, MomentsJacobian
% =============================================================================================

TimeStep = Model.TimeStep;

% Initial Parameters for optimization
% Must be set manually. But the fmnisearch optimization algorithm seems to be quite robust
switch Model.Name
    case 'CKLS'
        alpha = 0.5; 
        beta = -0.5;
        sigma = 0.5;
        gamma = 0.5;    
        a = alpha*TimeStep;
        b = beta*TimeStep + 1;
        InitialParams = [a b sigma gamma];
    case {'CIR', 'Vasicek'}            
        alpha = 0.5; 
        beta = -0.5;
        sigma = 0.5;            
        a = alpha*TimeStep;
        b = beta*TimeStep + 1;
        InitialParams = [a b sigma];               
end       

% ================ Firsr run, with identity weighting matrix===================================
W = eye(4);
options = optimset('LargeScale', 'off', 'MaxIter', 2500, 'MaxFunEvals', 3500, 'Display', Model.MatlabDisp, 'TolFun', 1e-40, 'TolX', 1e-40); 
[Params, Fval, Exitflag] =  fminsearch(@(Params) GMMobjective(Params, Model, W), InitialParams, options);   
switch Model.Name
    case 'CKLS'            
        Ralpha = Params(1)/TimeStep;
        Rbeta  = (Params(2)-1)/TimeStep;
        Rsigma2 = Params(3)^2;
        Rgamma = Params(4);
    case 'CIR'
        Ralpha = Params(1)/TimeStep;
        Rbeta  = (Params(2)-1)/TimeStep;
        Rsigma2 = Params(3)^2;
        Rgamma = 0.5;
    case 'Vasicek'
        Ralpha = Params(1)/TimeStep;
        Rbeta  = (Params(2)-1)/TimeStep;
        Rsigma2 = Params(3)^2;
        Rgamma = 0;        
end
if strcmp(Model.Disp, 'y')
    fprintf('\n Parameters etimates\n');
    fprintf(' First run without weighting matrix');
    fprintf('\n alpha  = %+3.5f\n beta   = %+3.5f\n sigma2 = %+3.5f\n gamma  = %+3.5f\n ------------------------------------- \n',...
    Ralpha, Rbeta, Rsigma2, Rgamma);
end

% ================= Second run, with optimal weighting matrix W ===============================
if Model.Iters > 0
    for i = 1 : Model.Iters
        InitialParams = Params;
        W = GMMweightsNW(Params, Model);        
        options = optimset('LargeScale', 'off', 'MaxIter', 2500, 'MaxFunEvals', 3500, 'Display', Model.MatlabDisp, 'TolFun', 1e-8, 'TolX', 1e-8); 
        [Params, Fval, Exitflag] =  fminsearch(@(Params) GMMobjective(Params, Model, W), InitialParams, options);   
        switch Model.Name
        case 'CKLS'            
            Ralpha = Params(1)/TimeStep;
            Rbeta  = (Params(2)-1)/TimeStep;
            Rsigma2 = Params(3)^2;
            Rgamma = Params(4);
        case 'CIR'
            Ralpha = Params(1)/TimeStep;
            Rbeta  = (Params(2)-1)/TimeStep;
            Rsigma2 = Params(3)^2;
            Rgamma = 0.5;
         case 'Vasicek'
            Ralpha = Params(1)/TimeStep;
            Rbeta  = (Params(2)-1)/TimeStep;
            Rsigma2 = Params(3)^2;
            Rgamma = 0;         
        end
        switch Model.Name
        case {'CIR', 'Vasicek'}
            %Chi2 statistics of the overidentified model. Are the empirical moments
            %sufficiently close to 0?
            Chi2statistic = Fval*length(Model.Data);
            Chi2pvalue = 1-chi2cdf(Chi2statistic, 1);   
            Results.Chi2statisitcs = Chi2statistic;
            Results.Chi2pvalue = Chi2pvalue;           
        end
        % t-statistic
        Nobs = length(Model.Data)-1;
        d = MomentsJacobian(Params, Model);
        VarParams = diag(inv(d'*W*d))./Nobs;
        Params(2) = Params(2)-1;
        Params(3) = Params(3)^2;
        Tstat = Params'./sqrt(VarParams);
        if strcmp(Model.Disp, 'y')
            fprintf('\n Parameters etimates, t-statistic in parentheses\n');
            fprintf(' Second run with weighting matrix, Iteration #%d\n', i);
            switch Model.Name
            case 'CKLS'
                fprintf('\n alpha  = %+3.5f (%+3.2f) \n beta   = %+3.5f (%+3.2f)\n sigma2 = %+3.5f (%+3.2f) \n gamma  = % +3.5f (%+3.2f)\n',...
                Ralpha, Tstat(1), Rbeta, Tstat(2), Rsigma2, Tstat(3), Rgamma, Tstat(4));
            case {'CIR', 'Vasicek'}
                fprintf('\n alpha  = %+3.5f (%+3.2f) \n beta   = %+3.5f (%+3.2f)\n sigma2 = %+3.5f (%+3.2f) \n gamma  = % +3.5f\n',...
                Ralpha, Tstat(1), Rbeta, Tstat(2), Rsigma2, Tstat(3), Rgamma);
                fprintf(' Chi2 statistic = %+2.4f\n', Chi2statistic);
                fprintf(' p-value        = %+2.4f\n', Chi2pvalue);
            end 
            fprintf(' Objective function = %2.3e\n', Fval);
            fprintf('---------------------------------------------- \n');            
        end
    end
Results.Tstat = Tstat;
Results.VarParams = VarParams;
end
Results.Params = [Ralpha Rbeta Rsigma2 Rgamma];
Results.Fval = Fval;
Results.Exitflag = Exitflag;
end