function viz_MSE_csxval_synth(MSE, S, Lambda, varargin)
% Visualization of Mean Squared Error curves with respect to the regularization 
% parameter estimation, on synthetic data
%
% Wouter Kouw
% Last update: 2017-04-24

% Add utility functions to path
addpath(genpath('../util'));

% Parse
p = inputParser;
addOptional(p, 'lineWidth', 3);
addOptional(p, 'fontSize', 18);
addOptional(p, 'markerSize', 12);
addOptional(p, 'colorMap', 'winter');
addOptional(p, 'title', '');
addOptional(p, 'save', false);
addOptional(p, 'saveName', 'viz_MSE_cvxval_synth');
parse(p, varargin{:});

% Number of target variance parameters
nS = length(S);

% Parse colormap
colorMap = eval(p.Results.colorMap);
ix = ceil(linspace(1,63,nS));

% Call figure
fg = figure();
hold on

% Loop over elements of the set of target variance parameters
sS = cell(nS,1);
for s = 1:nS
    
    % MSE curve for current target variance
    MSE_s = mean(MSE.Z(s,:,:),3);
    [minF,minL] = min(MSE_s);
    
    % Plot a Mean Squared Error curve for every target variance
    plot(Lambda, MSE_s, 'Color', colorMap(ix(s),:), 'LineWidth', p.Results.lineWidth);
    
    % Mark the minimum of the curve
    plot(Lambda(minL), minF, 'ks', 'MarkerSize', p.Results.markerSize, 'MarkerFaceColor', 'k');
    
    % Cast target variance to string
    sS{s} = ['\sigma_{Z} = ' num2str(sqrt(S(s)))];
end

% Add curves to legend
hh = findobj(gca, 'Type', 'Line');
legend(hh(end:-2:2), sS, 'Location', 'NorthEast');

% Set title
title(p.Results.title)

% Set axes descriptions
xlabel('\lambda');
ylabel('Mean Squared Error')
set(gca, 'YLim', [0 2]);
set(gca, 'FontSize', p.Results.fontSize, 'FontWeight', 'bold');
set(gcf, 'Position', [400 400 900 500], 'Color', 'w');

% Write to file
if p.Results.save
    disp(['Done. Writing to ' p.Results.saveName]);
    saveas(fg, p.Results.saveName);
end

end

