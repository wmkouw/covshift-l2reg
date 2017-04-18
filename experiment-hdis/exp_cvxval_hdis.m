% Script to run some covariate shift experiments
close all; clear all;
addpath(genpath('..\util'))

svnm = 'hdis';

% Importance weight estimator
iwT = 'nnew';

% Lambda range
Ll = linspace(-200,500,101);
nLl = length(Ll);

% Load data
dataname = 'hdis_imp0';
load(dataname)
prep = {'zscore'};
nR = 10;
D = D(:,[1:4 6:11]);

nD = length(domains)-1;
cc = [nchoosek(1:nD,2); fliplr(nchoosek(1:nD,2))];
ncc = size(cc,1);

% Optimizer
options.Method = 'lbfgs';
options.Display = 'full';
options.maxIter = 1e5;
options.xTol = 1e-12;
options.DerivativeCheck = 'off';

W = cell(ncc,nR);
MSE_V = zeros(ncc,nLl,nR);
dMSE_V = zeros(ncc,nLl,nR);
MSE_W = zeros(ncc,nLl,nR);
dMSE_W = zeros(ncc,nLl,nR);
MSE_Z = zeros(ncc,nLl,nR);
dMSE_Z = zeros(ncc,nLl,nR);
minl_X = zeros(ncc,nR);
minf_X = zeros(ncc,nR);
minl_W = zeros(ncc,nR);
minf_W = zeros(ncc,nR);
minl_Z = zeros(ncc,nR);
minf_Z = zeros(ncc,nR);

xl = [-5 5];
zl = [-5 5];
x = linspace( xl(1),xl(2),101);
z = linspace(-xl(2),xl(2),101);

for n = 1:ncc
    for r = 1:nR
        
        % Split out source and target
        ixX = domains(cc(n,1))+1:domains(cc(n,1)+1);
        ixZ = domains(cc(n,2))+1:domains(cc(n,2)+1);
        
        X = D(ixX,:);
        yX = y(ixX); yX(yX==2) = -1;
        NX = size(X,1);
        
        Z = D(ixZ,:);
        yZ = y(ixZ); yZ(yZ==2) = -1;
        NZ = size(Z,1);
        
        % Prepping
        X = da_prep(X', prep)';
        Z = da_prep(Z', prep)';
        
        % Split into training and validation
        ixT = randsample(1:NX,round(NX./2));
        ixV = setdiff(1:NX,ixT);
        T = X(ixT,:);
        yT = y(ixT);
        V = X(ixV,:);
        yV = y(ixV);
        
        % Augment data
        Xa = [X ones(size(X,1),1)];
        Ta = [T ones(size(T,1),1)];
        Va = [V ones(size(V,1),1)];
        Za = [Z ones(size(Z,1),1)];
        
        % Obtain importance weights
        switch lower(iwT)
            case 'none'
                W{n,r} = ones(1,size(V,1));
            case 'true'
                A = zeros(1,size(V,1));
                A(yV==-1) = pZy_n(V(yV==-1))./pXy_n(V(yV==-1));
                A(yV==1) = pZy_p(V(yV==1))./pXy_p(V(yV==1));
                W{n,r} = A;
            case 'gauss'
                W{n,r} = iw_Gauss(V',Z',0,realmax);
                W{n,r} = 1./W{n,r};
            case 'kmm'
                W{n,r} = iw_KMM(V',Z',0,realmax);
            case 'kliep'
                W{n,r} = iw_KLIEP(V,Z,0,realmax);
            case 'nnew'
                W{n,r} = iw_NNeW(V,Z,0,realmax, 'Laplace', 1);
            otherwise
                error(['Unknown importance weight estimator']);
        end
        W{n,r} = diag(W{n,r});
        
        % Obtain lambda-curve
        for l = 1:nLl
            [MSE_V(n,l,r),dMSE_V(n,l,r)] = cv_lambda_grad2(Ll(l),Ta,Va,yT,yV);
            [MSE_W(n,l,r),dMSE_W(n,l,r)] = cv_lambda_grad2(Ll(l),Ta,Va,yT,yV,W{n,r});
            [MSE_Z(n,l,r),dMSE_Z(n,l,r)] = cv_lambda_grad2(Ll(l),Ta,Za,yT,yZ);
        end
    
        %     % Run minimizer
        %     [minl(p),minf(p)] = minFunc(@cv_lambda_grad, 0, options, Xa,Za,yX,yX);

        [minf_X(n,r),minl_X(n,r)] = min(MSE_V(n,:,r));
        [minf_W(n,r),minl_W(n,r)] = min(MSE_W(n,:,r));
        [minf_Z(n,r),minl_Z(n,r)] = min(MSE_Z(n,:,r));
    end
end

di = 1; while exist(['results_iw-' iwT '_' svnm '_' num2str(di) '.mat'], 'file'); di = di+1; end
fn = ['results_iw-' iwT '_' svnm '_' num2str(di) '.mat'];
disp(['Done. Writing to ' fn]);
save(fn, 'minf_X', 'minl_X', 'minf_W', 'minl_W', 'minf_Z', 'minl_Z', 'W','MSE_W','dMSE_W','MSE_V','dMSE_V','MSE_Z','dMSE_Z');




