function [X_y0,X_y1,Z_y0,Z_y1,varargout] = gen_cvshift(varargin)
% For two class-conditional distributions, generate two different
% class-conditional distributions with equivalent class posteriors
% Optional input:
%       theta_Xy0:      Source class-conditional parameters, p(x|y=0;theta)
%       theta_Xy1:      Source class-conditional parameters, p(x|y=1;theta)
%       py:             Class priors p(y) (default: [1/2 1/2])
%       theta_Z:        Target data parameters p(z;theta)
%       N               Source sample size
%       M               Target sample size
%       ub              Upper bound for proposal distribution
% Output:
%       X_y0:           Samples from p(x|y=0)
%       X_y1:           Samples from p(x|y=1)
%       Z_y0:           Samples from p(z|y=0)
%       Z_y1:           Samples from p(z|y=1)
% Optional output:
%       pX_y0:          Source class-conditional, p(x|y=0)
%       pX_y1:          Source class-conditional, p(x|y=1)
%       pZ_y0:          Target class-conditional, p(z|y=0)
%       pZ_y1:          Target class-conditional, p(z|y=1)
%       py0_X:          Class posterior, p(y=0|x)
%       py1_X:          Class posterior, p(y=1|x)
%
% Copyright: Wouter M. Kouw
% Last update: 07-04-2016

% Parse hyperparameters
p = inputParser;
addOptional(p, 'xl', [-10 10]);
addOptional(p, 'zl', [-10 10]);
addOptional(p, 'N', 100);
addOptional(p, 'M', 200);
addOptional(p, 'ub', sqrt(2*pi));
addOptional(p, 'py', [1./2 1./2]);
addOptional(p, 'theta_Xy0', [0 1]);
addOptional(p, 'theta_Xy1', [1 1]);
addOptional(p, 'theta_Z', [0 2]);
addOptional(p, 'pZ', []);
parse(p, varargin{:});

%% Distribution functions

% Source class-conditional distributions
pX_y0 = @(x) pdf('Normal', x, p.Results.theta_Xy0(1), p.Results.theta_Xy0(2));
pX_y1 = @(x) pdf('Normal', x, p.Results.theta_Xy1(1), p.Results.theta_Xy1(2));

% Class-priors
py0 = p.Results.py(1);
py1 = p.Results.py(2);

% Class-posteriors
py0_X = @(x) pX_y0(x).*py0./(pX_y0(x).*py0 + pX_y1(x).*py1);
py1_X = @(x) pX_y1(x).*py1./(pX_y0(x).*py0 + pX_y1(x).*py1);

% Target data
if isempty(p.Results.pZ)
    pZ = @(z) pdf('Normal', z, p.Results.theta_Z(1), p.Results.theta_Z(2));
else
    pZ = p.Results.pZ;
end

% Target class-conditional distributions
pZ_y0 = @(z) (py0_X(z).*pZ(z)./py0);
pZ_y1 = @(z) (py1_X(z).*pZ(z)./py1);

% Use external rejection sampler
addpath(genpath('C:\Users\Wouter\Dropbox\Codes\sampleDist'))
X_y0 = sampleDist(pX_y0,p.Results.ub,round(p.Results.N.*py0),[p.Results.zl(1) p.Results.zl(2)], false);
X_y1 = sampleDist(pX_y1,p.Results.ub,round(p.Results.N.*py1),[p.Results.zl(1) p.Results.zl(2)], false);
Z_y0 = sampleDist(pZ_y0,p.Results.ub,round(p.Results.M.*py0),[p.Results.zl(1) p.Results.zl(2)], false);
Z_y1 = sampleDist(pZ_y1,p.Results.ub,round(p.Results.M.*py1),[p.Results.zl(1) p.Results.zl(2)], false);

if nargout > 4
    varargout{1} = pX_y0;
    varargout{2} = pX_y1;
    varargout{3} = pZ_y0;
    varargout{4} = pZ_y1;
    varargout{5} = py0_X;
    varargout{6} = py1_X;
end

end
