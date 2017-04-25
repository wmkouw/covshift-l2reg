% Script to run an experiment using synthetic data
close all;
clearvars;

% Run experiment
[MSE, minLambda, Lambda, S, W, params] = exp_cvxval_synth('S', [1 2 3 4].^2, 'Lambda', linspace(-100,500,101), 'nR', 10);

% Visualize synthetic data
viz_problem_cvxval_synth(S, params);

% Visualize MSE curves
viz_MSE_cvxval_synth(MSE, S, Lambda);
