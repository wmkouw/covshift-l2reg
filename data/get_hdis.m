function [D,y,domains,domain_names] = get_hdis(varargin)
% Script to download heart disease dataset from UCI machine learning repository

% Parse
p = inputParser;
addOptional(p, 'sav', false);
addOptional(p, 'impute', false);
parse(p, varargin{:});

%% Start downloading files

fprintf('Starting downloads..')
websave('processed.cleveland.data', ...
    'http://mlr.cs.umass.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data');

fprintf('.')
websave('processed.hungarian.data', ...
    'http://mlr.cs.umass.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data');

fprintf('.')
websave('processed.switzerland.data', ...
    'http://mlr.cs.umass.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data');

fprintf('.')
websave('processed.virginia.data', ...
    'http://mlr.cs.umass.edu/ml/machine-learning-databases/heart-disease/processed.va.data');

fprintf('Done \n')

%% Call parse scripts

parse_cleveland_gen()
parse_hungarian_gen()
parse_switzerland_gen()
parse_virginia_gen()

[D,y,domains,domain_names] = parse_hdis('sav', p.Results.sav, 'impute', p.Results.impute);

end
