% Script to run an experiment using heart disease data
close all;
clearvars;

% Define range of regularization parameters
Lambda = linspace(-100,500,101);

% Select importance-weight estimators
iwe = {'none', 'gauss', 'kliep', 'kmm', 'nnew'};
iwe_names = {'h_lV', 'w_G', 'w_kliep', 'w_kmm', 'w_nnew', 'h_lZ'};

% Loop over estimators
for i = 1:length(iwe)
    
    % Run experiment with current estimator
    [~,~,cc] = exp_csxval_hdis('Lambda', Lambda, 'nR', 10, 'iwe', iwe{i}, ...
        'save', true, 'saveName', ['results_csxval_hdis_' iwe{i}]);
end

% Gather results for all estimators
[results] = table_results_csxval_hdis(iwe, cc, 'iwe_names', iwe_names);

