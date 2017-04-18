function [iw] = iw_Gauss(X, Z, varargin)
% Uses two Gaussian distributions to estimate importance weights
% ! Do not use for high-dimensional data
% Function expects DxN matrices.

% Parse optionals
p = inputParser;
addOptional(p, 'lambda', 0);
addOptional(p, 'clip', realmax);
addOptional(p, 'viz', false);
parse(p, varargin{:});

% Shape
[~,D] = size(X);

% Calculate sparse Gaussian parameters
muX = mean(X,1);
muZ = mean(Z,1);
SX = cov(X)+p.Results.lambda*eye(D);
SZ = cov(Z)+p.Results.lambda*eye(D);

% Compute ratio of normal pdf
iw = mvnpdf(X, muZ, SZ) ./ mvnpdf(X, muX, SX); 

% Weight clipping
iw = min(p.Results.clip,max(0,iw));

if p.Results.viz
    figure()
    histogram(iw);
end

end
