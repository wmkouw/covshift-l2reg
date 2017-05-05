% Script to run an experiment using synthetic data
close all;
clearvars;

% Domain distribution parameters
params.X_yn = [-1 1];
params.X_yp = [ 1 1];

% Define target variances
S = [0.1 0.5 1.0 2.0 3.0 4.0].^2;

% Define range of regularization parameters
Lambda = linspace(0,500,101);

% Visualize problem setting
viz_problem_csxval_synth(S, params);

% Select importance-weight estimators
iwe = {'none', 'gauss', 'kliep', 'kmm', 'nnew', 'true'};
iwe_names = {'h_lV', 'w_G', 'w_kliep', 'w_kmm', 'w_nnew', 'pz_px', 'h_lZ'};


% Loop over estimators
for i = 1:length(iwe)
    
    % Run experiment with current estimator
    [MSE, ~, Lambda] = exp_csxval_synth('S', S, 'Lambda', Lambda, ...
        'theta_X_yn', params.X_yn, 'theta_X_yp', params.X_yp, ...
        'nR', 50, 'save', true, 'saveName', ['results_csxval_synth_' iwe{i}], ...
        'iwe', iwe{i});
    
    % Visualize MSE curves
    viz_MSE_csxval_synth(MSE, S, Lambda, 'title', iwe{i});
end

% Gather results for all estimators
[results] = table_results_csxval_synth(iwe, S, 'iwe_names', iwe_names);

