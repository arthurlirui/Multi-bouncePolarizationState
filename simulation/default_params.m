
%% Reflection Parameters
%params.refc = 0.2; % mirror-reflection coefficients
%params.refc_decay = 0.5; % mirror-reflection decay coefficients
params.refc = 0.0; % mirror-reflection coefficients
params.refc_decay = 0.0; % mirror-reflection decay coefficients
params.num_ref = 2; % the number of reflecetion in glass that we count
total_ref_coef = 0;
for i = 1:params.num_ref
    total_ref_coef = total_ref_coef + params.refc*params.refc_decay^(i-1);
end
params.mirror_ref_coef = total_ref_coef;
params.polar_ref_coef = 1-total_ref_coef;
params.n = 1.5;% air = 1
params.d = 5; % glass thickness 
params.thetalist = [10, 40, 60, 70, 80, 85];
params.philist = [0, 45, 90, 135];

params.subtask = 'partial';
if strcmp(params.subtask, 'weak_ref')
    params.ref_coef = 0.3;
    params.trans_coef = 1.0;
    params.ref_perp = 0.5;
    params.ref_para = 0.5;
    params.trans_perp = 0.5;
    params.trans_para = 0.5;
elseif strcmp(params.subtask, 'strong_ref')
    params.ref_coef = 1.3;
    params.trans_coef = 1.0;
    params.ref_perp = 0.5;
    params.ref_para = 0.5;
    params.trans_perp = 0.5;
    params.trans_para = 0.5;
elseif strcmp(params.subtask, 'partial')
    params.ref_coef = 1.0;
    params.trans_coef = 1.0;
    params.ref_perp = 0.7;
    params.ref_para = 0.3;
    params.trans_perp = 0.7;
    params.trans_para = 0.3;
else
end


params.rootpath = '/home/lir0b/Code/TransparenceDetection/src/pid/exp/paperdata';
params.savepath = fullfile(params.rootpath, params.subtask);
if ~exist(params.savepath), mkdir(params.savepath); end