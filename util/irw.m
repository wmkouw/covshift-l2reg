function [W,theta] = irw(X,Z,yX,varargin)
% Train an importance-weighted domain adaptive classifier
% Input:    X       source data (n x D samples)
%           Z       target data (m x D samples)
%           yX      source labels (n x 1)
% Optional:
%           l2      Additional l2-regularization parameters (default: 1e-3)
%           iw      Choice of importance weight estimator (default: 'logr')
%           loss    Choice of loss function (default: 'log')
%
% Output:   W       Trained linear classifier
%           theta   Found importance weights for source samples
%
% Wouter M. Kouw
% Last update: 22-12-2015

% Add dependencies to path
if isempty(which('minFunc')); error('Can not find minFunc'); end

% Parse optionals
p = inputParser;
addOptional(p, 'l2', 1e-3);
addOptional(p, 'iwe', 'logr');
addOptional(p, 'loss', 'logi');
parse(p, varargin{:});

% Check for solver
options.DerivativeCheck = 'off';
options.Method = 'lbfgs';
options.Display = 'final';

% Data shape
[n,D] = size(X);
uy = unique(yX)';
K = numel(uy);

% Check if labels are in [1,...,K]
if length(uy)==2 && all(uy==[-1 1] | uy==[0 1])
    yX(yX==-1 | yX==0) = 2;
elseif all(uy~=[1:K])
    error(['Labels should be a nx1 vector in [1,...,K]']); 
end

% Estimate importance weights
switch lower(p.Results.iwe)
    case 'gauss'
        theta = iwe_Gauss(X,Z,p.Results.l2);
    case 'kliep'
        theta = iwe_KLIEP(X,Z,p.Results.l2);
    case 'kmm'
        theta = iwe_KMM(X,Z, 1,'rbf');
    case 'nnew'
        theta = iwe_NNeW(X,Z);
end

switch p.Results.loss
    case 'quad'
        % Map nx1 vector in [1,...,K] to Kxn one-hot matrix
        if min(size(yX))==1
            Y = zeros(K,n);
            for i = 1:n; Y(yX(i),i) = 1; end
        else
            Y = yX;
        end
        
        % Closed-form solution to importance-weighted least-squares classifier
        bX = [bsxfun(@times, theta, X); ones(1,n)];
        W = (Y*bX'/ (bX*bX'+p.Results.l2*eye(D+1)))';
        
    case 'logi'
        % Minimize loss
        W_star = minFunc(@mWLR_grad, zeros((D+1)*K,1), options, X, yX, theta, p.Results.l2);
        
        % Output multiclass weight vector
        W = [reshape(W_star(1:end-K), [D K]); W_star(end-K+1:end)'];
        
    otherwise
        error('Loss function not implemented');
end

end

function [L, dL] = mWLR_grad(W,X,y,iw, lambda)
% Implementation of instance reweighted logistic regression
% Wouter Kouw
% 29-09-2014
% This function expects an 1xn label vector y with labels [1,..,K]

% Shape
[M,N] = size(X);
K = max(y);
W0 = reshape(W(M*K+1:end), [1 K]);
W = reshape(W(1:M*K), [M K]);

% Compute posterior
WX = bsxfun(@plus, W' * X, W0');
WX = exp(bsxfun(@minus, WX, max(WX, [], 1)));
WX = bsxfun(@rdivide, WX, max(sum(WX, 1), realmin));

% Negative log-likelihood of each sample
L = 0;
for i = 1:N
    L = L - iw(i)*log(max(WX(y(i), i),realmin));
end
L = L./N + lambda .* sum([W(:); W0(:)] .^ 2);

% Only compute gradient if requested
if nargout > 1
    
    % Compute positive part of gradient
    pos_E = zeros(M, K);
    pos_E0 = zeros(1, K);
    for k=1:K
        pos_E(:,k) = sum(bsxfun(@times, iw(y == k), X(:,y == k)), 2);
        pos_E0(k) = sum(iw(y == k));
    end
    
    % Compute negative part of gradient
    neg_E = bsxfun(@times, iw, X) * WX';
    neg_E0 = sum(bsxfun(@times, iw, WX), 2)';
    
    % Compute gradient
    dL = -1./N*[pos_E(:) - neg_E(:); pos_E0(:) - neg_E0(:)] + 2 .* lambda .* [W(:); W0(:)];
    
end
end
