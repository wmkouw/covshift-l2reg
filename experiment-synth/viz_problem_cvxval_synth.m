function viz_problem_cvxval_synth(S, params, varargin)
% Visualization of problem setting with synthetic data
%
% Wouter Kouw
% Last update: 2017-04-24

% Add utility functions to path
addpath(genpath('../util'));

% Parse
p = inputParser;
addOptional(p, 'domainLimits', [-5 5]);
addOptional(p, 'lineWidth', 3);
addOptional(p, 'fontSize', 18);
addOptional(p, 'markerSize', 12);
addOptional(p, 'save', false);
addOptional(p, 'saveName', 'viz/viz_problem_cvxval_synth');
parse(p, varargin{:});

% Number of target variance parameters
nS = length(S);

% Sweep domain limits
d = linspace(p.Results.domainLimits(1),p.Results.domainLimits(2), 201);

% Call figure
fg = figure();
hold on

% Plot source distributions
pX_yn = @(x) normpdf(x, params.X_yn(1), sqrt(params.X_yn(2)));
pX_yp = @(x) normpdf(x, params.X_yp(1), sqrt(params.X_yp(2)));
plot(d, pX_yn(d), 'b', 'LineWidth', p.Results.lineWidth+10);
plot(d, pX_yp(d), 'r', 'LineWidth', p.Results.lineWidth+10);

% Loop over elements of the set of target variance parameters
sS = cell(nS,1);
for s = 1:nS
    
	% Plot target distributions
    params.Z_yn = [-1 S(s)];
    params.Z_yp = [ 1 S(s)];
    pZ_yn = @(x) normpdf(x, params.Z_yn(1), sqrt(params.Z_yn(2)));
    pZ_yp = @(x) normpdf(x, params.Z_yp(1), sqrt(params.Z_yp(2)));
    plot(d, pZ_yp(d), 'Color', 'k', 'LineWidth', p.Results.lineWidth, 'LineStyle', '--');
    plot(d, pZ_yn(d), 'Color', 'k', 'LineWidth', p.Results.lineWidth, 'LineStyle', '--');
    
    % Cast target variance to string
    sS{s} = ['\sigma_{Z} = ' num2str(sqrt(S(s)))];
end

% Add target distributions to legend
hh = findobj(gca, 'Type', 'Line');
legend(hh(end-2:-2:1), sS, 'Location', 'northeast');

% Set axes descriptions
xlabel('x');
ylabel('p(x,y)');
set(gca, 'FontSize', p.Results.fontSize, 'FontWeight', 'bold');
set(fg, 'Color', 'w', 'Position', [400 400 900 500]);


% Write to file
if p.Results.save
    disp(['Done. Writing to ' p.Results.saveName]);
    saveas(fg, p.Results.saveName);
end

end

