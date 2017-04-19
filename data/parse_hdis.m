function [D,y,domains,domain_names] = parse_hdis(varargin)
% Script to join hospital data

% Parse hyperparameters
p = inputParser;
addOptional(p, 'sav', false);
addOptional(p, 'impute', false);
parse(p, varargin{:});

D = [];
y = [];
domains = [0];
domain_names = {'cleveland', 'hungary', 'switzerland', 'virginia'};

%% Cleveland
load hdis-cleveland

Y(Y>0) = 1;
y = [y; Y];
D = [D; X];
domains = [domains; size(D,1)];

%% Hungary
load hdis-hungarian

Y(Y>0) = 1;
y = [y; Y];
D = [D; X];
domains = [domains; size(D,1)];

%% Switzerland
load hdis-switzerland

Y(Y>0) = 1;
y = [y; Y];
D = [D; X];
domains = [domains; size(D,1)];

%% Virginia
load hdis-virginia

Y(Y>0) = 1;
y = [y; Y];
D = [D; X];
domains = [domains; size(D,1)];

%% Impute missing values

if p.Results.impute
    D(isnan(D)) = 0;
end

%% Write to matlab file

if p.Results.sav
    save('heart_disease.mat', 'D','y', 'domains', 'domain_names');
end

end


