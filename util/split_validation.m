function [T,yT,V,yV] = split_validation(X,y, varargin)
% Split dataset into mutually exclusive and exhaustive sets for training and
% validation.

% Parse
p = inputParser;
addOptional(p, 'proportion_val', 1/2);
parse(p, varargin{:});

% Inverse of proportion
iprop = round(1./p.Results.proportion_val);

% Size
[n,~] = size(X);

% Sample index set and its complement
ix = datasample(1:iprop, n);

% Training
T = X(ix~=1,:);
yT = y(ix~=1);

% Validation
V = X(ix==1,:);
yV = y(ix==1);


end
