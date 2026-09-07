% ================================================================
%  Forearm angular-amplitude perturbation data generation
%
%  Purpose
%     Quantify how a mis-estimated forearm swing amplitude — the bias observed
%     at the slowest walking speed (Section III-A) — propagates to the final
%     GRF estimate (Section III-B).
%
%  Design
%     In upper_body_model.m the forearm angle comes from integrating the
%     wrist gyro y-axis:
%         th_we_pred = cumtrapz(t, gy);  th_we_pred = th_we_pred - th_we_pred(1);
%     The amplitude is reduced by multiplying this angle by AMP. The upper
%     arm and torso follow at the same ratio because
%     th_es_pred = ES_SCALE * th_we_pred (Eq. 2). The angular velocity and
%     acceleration (om_we = gy, al_we) must be scaled by the same factor for
%     physical consistency, so gy itself is scaled as well.
%
%     OFFSET_MODE is 'manual', so the angular offsets are fixed constants —
%     only the amplitude changes and the amplitude effect is cleanly isolated.
%
%  Output (OUT_DIR)
%     stride_modeling_weamp_100.mat   AMP = 1.00  baseline
%     stride_modeling_weamp_090.mat   AMP = 0.90  amplitude -10%
%     stride_modeling_weamp_080.mat   AMP = 0.80  amplitude -20%
%     stride_modeling_weamp_070.mat   AMP = 0.70  amplitude -30%
%
%     NOTE: a fresh baseline (weamp_100) is generated so every level shares
%       identical generation settings; only AMP differs between the files.
%
%  Usage (MATLAB)
%     % load the kinematics into the workspace first, then
%     run('perturb_forearm_amplitude.m')
%     % afterwards: PERTURB_DIR=<OUT_DIR> python eval_amp_perturbation.py ...
% ================================================================

if ~exist('AMP_LIST','var'),  AMP_LIST = [1.00, 0.90, 0.80, 0.70]; end
if ~exist('BASE_DIR','var')
    % folder holding the merged kinematics .mat files
    BASE_DIR = fullfile('..', '..', '..', 'data');
end
if ~exist('OUT_DIR','var')
    % must match PERTURB_DIR of eval_amp_perturbation.py
    OUT_DIR = fullfile('..', '..', '..', 'data', 'perturb');
end
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

fprintf('=== forearm amplitude perturbation generation ===\n');
fprintf('out : %s\n', OUT_DIR);
fprintf('AMP : %s\n\n', mat2str(AMP_LIST));

% ---------------- load inputs (once) ----------------
if ~(exist('stride_kinematics_arm','var') && exist('stride_kinematics_arm_vector','var'))
    kin_file = fullfile(BASE_DIR, 'merged_stride_kinematics_arm_gyro.mat');
    vec_file = fullfile(BASE_DIR, 'merged_stride_kinematics_arm_vector_gyro.mat');
    if ~exist(kin_file,'file') || ~exist(vec_file,'file')
        error(['kinematics .mat files not found:\n  %s\n  %s\n' ...
               'If the paths differ, load stride_kinematics_arm / ' ...
               'stride_kinematics_arm_vector into the workspace first.'], kin_file, vec_file);
    end
    fprintf('[load] %s\n', kin_file);  Lk = load(kin_file);
    fprintf('[load] %s\n', vec_file);  Lv = load(vec_file);
    fn = fieldnames(Lk);  stride_kinematics_arm        = Lk.(fn{1});
    fn = fieldnames(Lv);  stride_kinematics_arm_vector = Lv.(fn{1});
    clear Lk Lv
    fprintf('[ok] kinematics loaded\n');
end

% upper_body_model.m sits next to this script in the repository.
this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir), this_dir = pwd; end
src = fileread(fullfile(this_dir, 'upper_body_model.m'));

% remove the trailing save so the patched runs do not litter the work folder
src = regexprep(src, "save\('stride_modeling','stride_modeling'\)", '');

% injection point: right after the gyro integration, before the ES/SC propagation
anchor = 'th_we_pred = th_we_pred - th_we_pred(1);';
assert(count(src, anchor) == 1, 'amplitude anchor is not unique.');

for a = AMP_LIST
    fprintf('\n-------- AMP = %.2f (%+.0f%%) --------\n', a, (a - 1) * 100);

    if a == 1.0
        patched = src;                       % baseline: no injection
    else
        % scale the angle and the angular velocity by the same factor
        inject  = sprintf(['%s th_we_pred = %.6f * th_we_pred; ' ...
                           'gy = %.6f * gy;'], anchor, a, a);
        patched = strrep(src, anchor, inject);
    end

    tmp = fullfile(tempdir, sprintf('ubm_weamp_%03d.m', round(a * 100)));
    fid = fopen(tmp, 'w');  fwrite(fid, patched);  fclose(fid);

    old = cd(tempdir);
    try
        run(tmp);
    catch ME
        cd(old);  rethrow(ME);
    end
    cd(old);

    out = fullfile(OUT_DIR, sprintf('stride_modeling_weamp_%03d.mat', round(a * 100)));
    save(out, 'stride_modeling', '-v7.3');    % the Python loader requires v7.3
    fprintf('[save] %s\n', out);
    clear stride_modeling
end

fprintf('\n=== done ===\n');
fprintf('next step: inference with eval_amp_perturbation.py (no retraining)\n');
