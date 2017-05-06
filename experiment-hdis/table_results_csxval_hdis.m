function [T] = table_results_csxval_hdis(iwe, cc, varargin)
% Summarization of results on regularization parameter estimation under
% covariate shift, on the heart disease data from the UCI ML repo.
%
% Wouter Kouw
% Pattern Recognition Laboratory, TU Delft
% Last update: 2017-04-24

% Add utility functions to path
addpath(genpath('../util'));

% Parse
p = inputParser;
addOptional(p, 'iwe_names', '');
addOptional(p, 'save', false);
addOptional(p, 'saveName', 'results_csxval_synth_');
addOptional(p, 'domainNames', {'C','V','H','S'});
parse(p, varargin{:});

% Number of combinations
nC = size(cc,1);

% Number of estimators
nE = length(iwe);

% Preallocate
lambda_hat.V = zeros(nE,nC);
lambda_hat.W = zeros(nE,nC);
lambda_hat.Z = zeros(nE,nC);
for e = 1:nE
    
    fn = [p.Results.saveName iwe{e} '.mat'];
    load(fn);
        
    % Collect optimal lambda 
    lambda_hat.V(e,:) = mean(minLambda.V,2);
    lambda_hat.W(e,:) = mean(minLambda.W,2);
    lambda_hat.Z(e,:) = mean(minLambda.Z,2);
end

% Turn combinations into readable row names
rowNames = cell(nC,1);
for c = 1:nC
       rowNames{c} = [p.Results.domainNames{cc(c,1)} '_' p.Results.domainNames{cc(c,2)}];
end

% Summarize to table
T = array2table([lambda_hat.W' lambda_hat.Z(1,:)'], ...
    'RowNames',  rowNames, ...
    'VariableNames', p.Results.iwe_names); 

end

