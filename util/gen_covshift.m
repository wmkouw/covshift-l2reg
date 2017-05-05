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
addOptional(p, 'ubX', 1./sqrt(2*pi));
addOptional(p, 'ubZ', 1./sqrt(2*pi));
addOptional(p, 'py', [1./2 1./2]);
addOptional(p, 'theta_Xyn', [-1 1]);
addOptional(p, 'theta_Xyp', [ 1 1]);
parse(p, varargin{:});

%% Distribution functions

% Source class-conditional distributions
pX_yn = @(x) normpdf(x, p.Results.theta_Xyn(1), sqrt(p.Results.theta_Xyn(2)));
pX_yp = @(x) normpdf(x, p.Results.theta_Xyp(1), sqrt(p.Results.theta_Xyp(2)));

% Class-priors
pyn = p.Results.py(1);
pyp = p.Results.py(2);

% Class-posteriors
pyn_X = @(x) pX_yn(x).*pyn./(pX_yn(x).*pyn + pX_yp(x).*pyp);
pyp_X = @(x) pX_yp(x).*pyp./(pX_yn(x).*pyn + pX_yp(x).*pyp);

% Target class-conditional distributions
pZ_yn = @(z) (pyn_X(z).*pZ(z)./pyn);
pZ_yp = @(z) (pyp_X(z).*pZ(z)./pyp);

% Use external rejection sampler
X_yn = sampleDist(pX_yn,p.Results.ubX,round(p.Results.N.*pyn),[p.Results.xl(1) p.Results.xl(2)], false);
X_yp = sampleDist(pX_yp,p.Results.ubX,round(p.Results.N.*pyp),[p.Results.xl(1) p.Results.xl(2)], false);
Z_yn = sampleDist(pZ_yn,p.Results.ubZ,round(p.Results.M.*pyn),[p.Results.zl(1) p.Results.zl(2)], false);
Z_yp = sampleDist(pZ_yp,p.Results.ubZ,round(p.Results.M.*pyp),[p.Results.zl(1) p.Results.zl(2)], false);

if nargout > 4
    varargout{1} = pX_yn;
    varargout{2} = pX_yp;
    varargout{3} = pZ_yn;
    varargout{4} = pZ_yp;
    varargout{5} = pyn_X;
    varargout{6} = pyp_X;
end

end
