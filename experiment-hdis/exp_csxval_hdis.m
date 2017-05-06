function [MSE,minLambda,cc,W] = exp_csxval_hdis(varargin)
% Experiment with regularization parameter estimation on the heart disease
% dataset from the UCI machine learning repository.
%
% Wouter Kouw
% Pattern Recognition Laboratory, TU Delft
% Last update: 2017-04-24

% Add utility functions to path
addpath(genpath('../util'));
if isempty(which('sampleDist')); error('Please add sampleDist to the addpath'); end

% Parse
p = inputParser;
addOptional(p, 'Lambda', linspace(-100,500,101));
addOptional(p, 'nR', 10);
addOptional(p, 'prep', {''});
addOptional(p, 'save', false);
addOptional(p, 'saveName', 'results_exp_cvxval_hdis');
addOptional(p, 'iwe', 'true');
parse(p, varargin{:});

% Lambda range
Lambda = p.Results.Lambda;
nL = length(Lambda);

% Load data
try
    load('../data/hdis/heart_disease.mat')
catch
    cd '../data/hdis/'
    [D,y,domains] = get_hdis('save', true, 'impute', true);
    cd '../../experiment-hdis/'
end

% Map other label to negative
y(y~=1) = -1;

% Number of features
[~,nF] = size(D);

% Number of domains
nD = length(domains)-1;

% Create pairwise combinations between domains
cc = [nchoosek(1:nD,2); fliplr(nchoosek(1:nD,2))];
nC = size(cc,1);

% Preallocate variables
W = cell(nD,p.Results.nR);
MSE.V = zeros(nD,nL,p.Results.nR);
MSE.W = zeros(nD,nL,p.Results.nR);
MSE.Z = zeros(nD,nL,p.Results.nR);
minl.V = zeros(nD,p.Results.nR);
minl.W = zeros(nD,p.Results.nR);
minl.Z = zeros(nD,p.Results.nR);

for r = 1:p.Results.nR
    % Report progress
    fprintf('Running repeat %d / %d \n', r, p.Results.nR);
    for c = 1:nC
        
        % Find domain indices
        ixX = domains(cc(c,1))+1:domains(cc(c,1)+1);
        ixZ = domains(cc(c,2))+1:domains(cc(c,2)+1);
        
        % Slice out source domain
        X = D(ixX,:);
        yX = y(ixX);
        
        % Slice out target domain
        Z = D(ixZ,:);
        yZ = y(ixZ);
        
        % Preprocess data
        X = da_prep(X', p.Results.prep)';
        Z = da_prep(Z', p.Results.prep)';
        
        % Split into training and validation
        [T,yT, V,yV] = split_validation(X,yX);
        
        % Augment data
        Ta = [T ones(size(T,1),1)];
        Va = [V ones(size(V,1),1)];
        Za = [Z ones(size(Z,1),1)];
        
        % Obtain importance weights
        switch lower(p.Results.iwe)
            case 'none'
                W{c,r} = ones(size(V,1),1);
            case 'gauss'
                W{c,r} = iwe_Gauss(V,Z, 'lambda', 1e-5);
            case 'kmm'
                W{c,r} = iwe_KMM(V,Z, 'theta', 1, 'B', 1000);
            case 'kliep'
                W{c,r} = iwe_KLIEP(V,Z, 'sigma', 0);
            case 'nnew'
                W{c,r} = iwe_NNeW(V,Z, 'Laplace', 1);
            otherwise
                error(['Unknown importance weight estimator']);
        end
        
        % Loop over regularization parameter values
        for l = 1:nL
            
            % Analytical solution to regularized least-squares classifier
            theta = (Ta'*Ta + Lambda(l)*eye(nF+1))\Ta'*yT;
            
            % Mean squared error curves
            MSE.V(c,l,r) = mean((Va*theta - yV).^2,1);
            MSE.W(c,l,r) = mean((Va*theta - yV).^2.*W{c,r},1);
            MSE.Z(c,l,r) = mean((Za*theta - yZ).^2,1);
        end
        
        % Minima of mean squared error curves
        [~,minl.V(c,r)] = min(MSE.V(c,:,r), [], 2);
        [~,minl.W(c,r)] = min(MSE.W(c,:,r), [], 2);
        [~,minl.Z(c,r)] = min(MSE.Z(c,:,r), [], 2);
    end
end

% Arg min of mean squared error curves
minLambda.V = Lambda(minl.V);
minLambda.W = Lambda(minl.W);
minLambda.Z = Lambda(minl.Z);

% Write to file
if p.Results.save
    fn = [p.Results.saveName '.mat'];
    disp(['Done. Writing to ' fn]);
    save(fn, 'MSE', 'minLambda', 'Lambda', 'theta', 'W');
end


end
