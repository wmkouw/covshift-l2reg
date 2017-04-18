% Script to run some covariate shift experiments
close all; clear all;
addpath(genpath('..\util'))

% True distribution Parameters
N = 1e2;
M = 1e2;
theta_Xy0 = [-1 1];
theta_Xy1 = [ 1 1];
py0 = 1./2;
py1 = 1./2;
savnm = 'results\varshift';

% Importance weight estimator
iwT = 'gauss';

% Lambda range
Lambda = linspace(-200,500,101);
nLl = length(Lambda);

% Variances
P = [0.1 0.5 1 2 3 4];
nP = length(P);

% Number of repeats
nR = 100;

% Optimizer
options.Method = 'lbfgs';
options.Display = 'full';
options.maxIter = 1e4;
options.xTol = 1e-7;
options.DerivativeCheck = 'off';

W = cell(nP,nR);
MSE_X = zeros(nP,nLl,nR);
dMSE_X = zeros(nP,nLl,nR);
MSE_W = zeros(nP,nLl,nR);
dMSE_W = zeros(nP,nLl,nR);
MSE_Z = zeros(nP,nLl,nR);
dMSE_Z = zeros(nP,nLl,nR);
minl_X = zeros(nP,nR);
minf_X = zeros(nP,nR);
minl_W = zeros(nP,nR);
minf_W = zeros(nP,nR);
minl_Z = zeros(nP,nR);
minf_Z = zeros(nP,nR);

xl = [-5 5];
zl = [-5 5];
x = linspace( xl(1),xl(2),101);
z = linspace(-xl(2),xl(2),101);

for p = 1:nP
    for r = 1:nR
        
        % Target variance parameters
        theta_Zy0 = [-1 P(p)];
        theta_Zy1 = [ 1 P(p)];
        pZ_y0 = @(z) pdf('Normal', z, theta_Zy0(1), theta_Zy0(2));
        pZ_y1 = @(z) pdf('Normal', z, theta_Zy1(1), theta_Zy1(2));
        pZ = @(z) (pZ_y0(z).*py0 + pZ_y1(z).*py1);
        
        % Generate class-conditional distributions and sample sets
        [Xy_p,Xy_n,Zy_p,Zy_n,pXy_p,pXy_n,pZy_p,pZy_n] = gen_cvshift('theta_Xy0', theta_Xy0,'theta_Xy1', theta_Xy1, 'N', N, 'M', M, 'pZ', pZ);
        
        Vy_n = sampleDist(pXy_n,1.1./sqrt(2*pi),round(N.*py0),[zl(1) zl(2)], false);
        Vy_p = sampleDist(pXy_p,1.1./sqrt(2*pi),round(N.*py1),[zl(1) zl(2)], false);
        
        % Concatenate to dataset
        X = [Xy_n; Xy_p];
        V = [Vy_n; Vy_p];
        Z = [Zy_n; Zy_p];
        yX = [-ones(size(Xy_n,1),1); ones(size(Xy_p,1),1)];
        yV = [-ones(size(Vy_n,1),1); ones(size(Vy_p,1),1)];
        yZ = [-ones(size(Xy_n,1),1); ones(size(Xy_p,1),1)];
        
        % Obtain importance weights
        switch lower(iwT)
            case 'none'
                W{p,r} = ones(1,size(V,1));
            case 'true'
                A = zeros(1,size(V,1));
                A(yV==-1) = pZy_n(V(yV==-1))./pXy_n(V(yV==-1));
                A(yV==1) = pZy_p(V(yV==1))./pXy_p(V(yV==1));
                W{p,r} = A;
            case 'cauchy'
                W{p,r} = iw_Cauchy(V,Z,0,realmax);
            case 'gauss'
                W{p,r} = iw_Gauss(V',Z',0,realmax);
            case 'kmm'
                W{p,r} = iw_KMM(V',Z',0,realmax);
            case 'kliep'
                W{p,r} = iw_KLIEP(V,Z,0,realmax);
            case 'nnew'
                W{p,r} = iw_NNeW(V,Z,0,realmax, 'Laplace', 1);
            otherwise
                error(['Unknown importance weight estimator']);
        end
        W{p,r} = diag(W{p,r});
        
        % Augment data
        Xa = [X ones(size(X,1),1)];
        Va = [V ones(size(V,1),1)];
        Za = [Z ones(size(Z,1),1)];
        
        % Obtain lambda-curve
        for l = 1:nLl
            [MSE_X(p,l,r),dMSE_X(p,l,r)] = cv_lambda_grad2(Lambda(l),Xa,Va,yX,yV);
            [MSE_W(p,l,r),dMSE_W(p,l,r)] = cv_lambda_grad2(Lambda(l),Xa,Va,yX,yV,W{p,r});
            [MSE_Z(p,l,r),dMSE_Z(p,l,r)] = cv_lambda_grad2(Lambda(l),Xa,Za,yX,yZ);
        end

        % Find minima
        [minf_X(p,r),minl_X(p,r)] = min(MSE_X(p,:,r));
        [minf_W(p,r),minl_W(p,r)] = min(MSE_W(p,:,r));
        [minf_Z(p,r),minl_Z(p,r)] = min(MSE_Z(p,:,r));
    end
end

minLa_X = Lambda(minl_X);
minLa_W = Lambda(minl_W);
minLa_Z = Lambda(minl_Z);

di = 1; while exist(['results_iw-' iwT '_' savnm '_' num2str(di) '.mat'], 'file'); di = di+1; end
fn = [savnm '_iw-' iwT '_' num2str(di) '.mat'];
disp(['Done. Writing to ' fn]);
save(fn, 'minLa_X', 'minLa_W', 'minLa_Z', 'minf_X', 'minl_X', 'minf_W', 'minl_W', 'minf_Z', 'minl_Z');




