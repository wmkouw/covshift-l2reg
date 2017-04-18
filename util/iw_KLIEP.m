function [iw] = iw_KLIEP(X,Z,varargin)
% Kullback-Leibler Importance Estimation Procedure
%
% Sugiyama, Nakajima, Kashima, von Bunau & Kawanabe. Direct Importance
% Estimation with Model Selection and Its Application to Covariate Shift
% (NIPS, 2008).

% Parse optionals
p = inputParser;
addOptional(p, 'nK',100);
addOptional(p, 'sigma', 0);
addOptional(p, 'fold',5);
addOptional(p, 'clip', Inf);
parse(p, varargin{:});

% Shape
[N,D]=size(X);
[M,~]=size(Z);

%%%%%%%%%%%%%%%% Choosing Gaussian kernel center `x_ce'
rand_index=randperm(M);
nK=min(p.Results.nK,M);
x_ce=Z(rand_index(1:nK),:);

if p.Results.sigma==0;
    %%%%%%%%%%%%%%%% Searching Gaussian kernel width `sigma_chosen'
    sigma=10; score=-inf;
    
    for epsilon=log10(sigma)-1:-1:-5
        for iteration=1:9
            sigma_new=sigma-10^epsilon;
            
            cv_index=randperm(M);
            cv_split=floor([0:M-1]*p.Results.fold./M)+1;
            score_new=0;
            
            X_de=kernel_Gaussian(X,x_ce,sigma_new);
            X_nu=kernel_Gaussian(Z,x_ce,sigma_new);
            mean_X_de=mean(X_de,1)';
            for i=1:p.Results.fold
                alpha_cv=KLIEP_learning(mean_X_de,X_nu(cv_index(cv_split~=i),:));
                wh_cv=X_nu(cv_index(cv_split==i),:)*alpha_cv;
                score_new=score_new+mean(log(wh_cv))/p.Results.fold;
            end
            
            if (score_new-score)<=0
                break
            end
            score=score_new;
            sigma=sigma_new;
            disp(sprintf('  score=%g,  sigma=%g',score,sigma))
        end %iteration
    end %epsilon
else
    sigma = p.Results.sigma;
end
disp(sprintf('sigma = %g',sigma))


%%%%%%%%%%%%%%%% Computing the final solution 'iw'
X_de=kernel_Gaussian(X,x_ce,sigma);
X_nu=kernel_Gaussian(Z,x_ce,sigma);
mean_X_de=mean(X_de,1)';
alphah=KLIEP_learning(mean_X_de,X_nu);
iw=(X_de*alphah)';

% Weight clipping
iw = min(p.Results.clip,max(0,iw));

end

function px=pdf_Gaussian(x,mu,sigma)

[d,nx]=size(x);

tmp=(x-repmat(mu,[1 nx]))./repmat(sigma,[1 nx])/sqrt(2);
px=(2*pi)^(-d/2)/prod(sigma)*exp(-sum(tmp.^2,1));

end

function X=kernel_Gaussian(x,c,sigma)

[nx d]=size(x);
[nc d]=size(c);

distance2=repmat(sum(c.^2,2), [1 nx])'+repmat(sum(x.^2,2)', [nc 1])'-2*x*c';
X=exp(-distance2/(2*sigma^2));

end

function [alpha,Xte_alpha,score]=KLIEP_projection(alpha,Xte,b,c)
%  alpha=alpha+b*(1-sum(b.*alpha))/c;
alpha=alpha+b*(1-sum(b.*alpha))*pinv(c,10^(-20));
alpha=max(0,alpha);
%  alpha=alpha/sum(b.*alpha);
alpha=alpha*pinv(sum(b.*alpha),10^(-20));
Xte_alpha=Xte*alpha;
score=mean(log(Xte_alpha));

end

function [alpha,score]=KLIEP_learning(mean_X_de,X_nu)

[n_nu,nc]=size(X_nu);

max_iteration=100;
epsilon_list=10.^[3:-1:-3];
c=sum(mean_X_de.^2);
alpha=ones(nc,1);
[alpha,X_nu_alpha,score]=KLIEP_projection(alpha,X_nu,mean_X_de,c);

for epsilon=epsilon_list
    for iteration=1:max_iteration
        alpha_tmp=alpha+epsilon*X_nu'*(1./X_nu_alpha);
        [alpha_new,X_nu_alpha_new,score_new]=...
            KLIEP_projection(alpha_tmp,X_nu,mean_X_de,c);
        if (score_new-score)<=0
            break
        end
        score=score_new;
        alpha=alpha_new;
        X_nu_alpha=X_nu_alpha_new;
    end
end

end
