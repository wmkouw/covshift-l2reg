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


%% Samples

% Source samples
% X_y0 = randn(p.Results.N./2,1)*sqrt(p.Results.theta_Xy0(2)) + p.Results.theta_Xy0(1);
% X_y1 = randn(p.Results.N./2,1)*sqrt(p.Results.theta_Xy1(2)) + p.Results.theta_Xy1(1);
%
% % Target samples through rejection sampling
% % Dominating distribution
% pD = @(x) pdf('Normal', x, (p.Results.theta_Xy0(1)+p.Results.theta_Xy1(1))./2, (p.Results.theta_Xy0(2)+p.Results.theta_Xy1(2)));
%
% % Constant bound on the likelihood ratio
% M = 20;
%
% % Start sampling for p(z|y=0)
% Z_y0 = zeros(p.Results.M,1);
% j = 1;
% while j < p.Results.M+1
% u = rand(1);
% z = rand(1)*(p.Results.zl(2)-p.Results.zl(1)) + p.Results.zl(1);
% if u < (pZ_y0(z)./(M*pD(z)));
%     Z_y0(j) = z;
%     j = j + 1;
%     if rem(u,101)==1; disp('.'); end
% end
% end
%
% % Start sampling for p(z|y=1)
% Z_y1 = zeros(p.Results.M,1);
% j = 1;
% while j < p.Results.M+1
%     u = rand(1);
%     z = rand(1)*(p.Results.zl(2)-p.Results.zl(1)) + p.Results.zl(1);
%     if u < (pZ_y1(z)./(M*pD(z)));
%         Z_y1(j) = z;
%         j = j + 1;
%     end
% end

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

function X = sampleDist(f,M,N,b,mkplt)
% SAMPLEDIST  Sample from an arbitrary distribution
%     sampleDist(f,M,N,b) retruns an array of size X of random values
%     sampled from the distribution defined by the probability density
%     function refered to by handle f, over the range b = [min, max].
%     M is the threshold value for the proposal distribution, such that
%     f(x) < M for all x in b.
%
%     sampleDist(...,true) also generates a histogram of the results
%     with an overlay of the true pdf.
%
%     Examples:
%     %Sample from a step function over [0,1]:
%     X = sampleDist(@(x)1.3*(x>=0&x<0.7)+0.3*(x>=0.7&x<=1),...
%                    1.3,1e6,[0,1],true);
%     %Sample from a normal distribution over [-5,5]:
%     X = sampleDist(@(x) 1/sqrt(2*pi) *exp(-x.^2/2),...
%                    1/sqrt(2*pi),1e6,[-5,5],true);
%

% Dmitry Savransky (dsavrans@princeton.edu)
% May 11, 2010

n = 0;
X = zeros(N,1);
counter = 0;

while 1
    while n < N && counter < 1e6
        x = b(1) + rand(2*N,1)*diff(b);
        uM = M*rand(2*N,1);
        x = x(uM < f(x));
        X(n+1:min([n+length(x),N])) = x(1:min([length(x),N - n]));
        n = n + length(x);
        counter = counter+1;
    end
    if ~isempty(x)
        break;
    end
end

if exist('mkplt','var') && mkplt
    nsamps = max([min([N/20,1e4]),10]);
    [n,c] = hist(X,nsamps);
    bar(c,n/N*nsamps/(max(X) - min(X)))
    hold on
    plot(c,f(c),'r','LineWidth',2)
    hold off
end

end