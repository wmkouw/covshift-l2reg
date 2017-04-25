function [X_yn, X_yp, Z_yn, Z_yp, varargout] = gen_covshift(pZ,varargin)
% Generate synthetic covariate shift data:
% For two class-conditional distributions, generate two different
% class-conditional distributions with equivalent class posteriors
%
% Input:
%       pZ              Target data distribution
% Optional input:
%       theta_Xyn:      Source class-conditional parameters, p(x|y= -1;theta)
%       theta_Xyp:      Source class-conditional parameters, p(x|y= +1;theta)
%       py:             Class priors p(y) (default: [1/2 1/2])
%       N               Source sample size
%       M               Target sample size
%       ub              Upper bound for proposal distribution
% Output:
%       X_yn:           Samples from p(x|y= -1)
%       X_yp:           Samples from p(x|y= +1)
%       Z_yn:           Samples from p(z|y= -1)
%       Z_yp:           Samples from p(z|y= +1)
% Optional output:
%       pX_yn:          Source class-conditional, p(x|y= -1)
%       pX_yp:          Source class-conditional, p(x|y= +1)
%       pZ_yn:          Target class-conditional, p(z|y= -1)
%       pZ_yp:          Target class-conditional, p(z|y= +1)
%       pyn_X:          Class posterior, p(y= -1|x)
%       pyp_X:          Class posterior, p(y= +1|x)
%
% Copyright: Wouter M. Kouw
% Last update: 07-04-2016

% Check for external rejection sampler
if isempty(which('sampleDist')); error('Please add sampleDist to the addpath'); end

% Parse hyperparameters
p = inputParser;
addOptional(p, 'xl', [-10 10]);
addOptional(p, 'zl', [-10 10]);
addOptional(p, 'N', 100);
addOptional(p, 'M', 100);
addOptional(p, 'ub', 1./sqrt(2*pi));
addOptional(p, 'py', [1./2 1./2]);
addOptional(p, 'theta_Xyn', [-1 1]);
addOptional(p, 'theta_Xyp', [ 1 1]);
parse(p, varargin{:});

%% Distribution functions

% Source class-conditional distributions
pX_yn = @(x) normpdf(x, p.Results.theta_Xy0(1), sqrt(p.Results.theta_Xy0(2)));
pX_yp = @(x) normpdf(x, p.Results.theta_Xy1(1), sqrt(p.Results.theta_Xy1(2)));

% Class-priors
py0 = p.Results.py(1);
py1 = p.Results.py(2);

% Class-posteriors
py0_X = @(x) pX_yn(x).*py0./(pX_yn(x).*py0 + pX_yp(x).*py1);
py1_X = @(x) pX_yp(x).*py1./(pX_yn(x).*py0 + pX_yp(x).*py1);

% Target class-conditional distributions
pZ_y0 = @(z) (py0_X(z).*pZ(z)./py0);
pZ_y1 = @(z) (py1_X(z).*pZ(z)./py1);

% Use external rejection sampler
X_yn = sampleDist(pX_yn,p.Results.ub,round(p.Results.N.*py0),[p.Results.zl(1) p.Results.zl(2)], false);
X_yp = sampleDist(pX_yp,p.Results.ub,round(p.Results.N.*py1),[p.Results.zl(1) p.Results.zl(2)], false);
Z_yn = sampleDist(pZ_y0,p.Results.ub,round(p.Results.M.*py0),[p.Results.zl(1) p.Results.zl(2)], false);
Z_yp = sampleDist(pZ_y1,p.Results.ub,round(p.Results.M.*py1),[p.Results.zl(1) p.Results.zl(2)], false);

if nargout > 4
    varargout{1} = pX_yn;
    varargout{2} = pX_yp;
    varargout{3} = pZ_yn;
    varargout{4} = pZ_yp;
    varargout{5} = pyn_X;
    varargout{6} = pyp_X;
end

end
