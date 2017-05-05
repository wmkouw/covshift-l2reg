function [T] = table_results_csxval_synth(iwe, S, varargin)
% Summarization of results on regularization parameter estimation under
% covariate shift, on synthetic data
%
% Wouter Kouw
% Last update: 2017-04-24

% Add utility functions to path
addpath(genpath('../util'));

% Parse
p = inputParser;
addOptional(p, 'iwe_names', '');
addOptional(p, 'save', false);
addOptional(p, 'saveName', 'results_csxval_synth_');
parse(p, varargin{:});

% Number of target variances
nS = length(S);

% Number of estimators
nE = length(iwe);

% Preallocate
lambda_hat.V = zeros(nE,nS);
lambda_hat.W = zeros(nE,nS);
lambda_hat.Z = zeros(nE,nS);
for e = 1:nE
    
    fn = [p.Results.saveName iwe{e} '.mat'];
    load(fn);
        
    % Collect optimal lambda 
    lambda_hat.V(e,:) = mean(minLambda.V,2);
    lambda_hat.W(e,:) = mean(minLambda.W,2);
    lambda_hat.Z(e,:) = mean(minLambda.Z,2);
end

% Summarize to table
T = array2table([lambda_hat.W' lambda_hat.Z(1,:)'], ...
    'RowNames',  arrayfun(@(x) sprintf('%.1f', x), S, 'unif',0), ...
    'VariableNames', p.Results.iwe_names); 

end

