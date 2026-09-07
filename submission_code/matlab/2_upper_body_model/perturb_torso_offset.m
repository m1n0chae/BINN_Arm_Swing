% ================================================================
%  Torso angular-offset perturbation data generation
%
%  Purpose
%     Quantify how far a systematic marker-placement bias in the torso
%     angular offset propagates to the final GRF estimate (Section III-B).
%
%  Design
%     stride_modeling is regenerated with a constant added to the angular
%     offsets of upper_body_model.m. The trained model is left untouched;
%     only the inference input changes, isolating the sensitivity.
%
%     Because each segment has a different sensitivity, they are perturbed
%     SEPARATELY:
%          SEG='we'   forearm (wrist-elbow)
%          SEG='es'   upper arm (elbow-shoulder)
%          SEG='sc'   torso (shoulder-sacrum)
%          SEG='all'  all three at once (for reference)
%
%  Eq. (3): th_i_fd = th_i_pred + off_i. Adding Δ to the offset rotates that
%  segment's contribution vector by exactly Δ (rddx' = cosΔ·rddx − sinΔ·rddz).
%
%  Usage (MATLAB) — the setting reported in Section III-B of the paper
%     BASE_DIR='<folder with the merged kinematics .mat files>';
%     OUT_DIR ='<PERTURB_DIR of eval_offset_perturbation.py>';
%     SEG='sc'; DELTA_LIST=[-30,-15,15,30]; run('perturb_torso_offset.m')
%     SEG='none';                           run('perturb_torso_offset.m')  % baseline
% ================================================================

if ~exist('SEG','var'),        SEG = 'sc'; end
if ~exist('BASE_DIR','var')
    % Point this at the folder holding the merged kinematics .mat files.
    BASE_DIR = fullfile('..', '..', '..', 'data');
end
if ~exist('OUT_DIR','var')
    % Must match PERTURB_DIR of eval_offset_perturbation.py.
    OUT_DIR = fullfile('..', '..', '..', 'data', 'perturb');
end
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

fprintf('=== torso offset perturbation generation ===\n');
fprintf('SEG=%s\n', SEG);

% ---------------- load inputs (once) ----------------
if ~(exist('stride_kinematics_arm','var') && exist('stride_kinematics_arm_vector','var'))
    kin_file = fullfile(BASE_DIR, 'merged_stride_kinematics_arm_gyro.mat');
    vec_file = fullfile(BASE_DIR, 'merged_stride_kinematics_arm_vector_gyro.mat');
    fprintf('[load] %s\n', kin_file);  Lk = load(kin_file);
    fprintf('[load] %s\n', vec_file);  Lv = load(vec_file);
    fn = fieldnames(Lk);  stride_kinematics_arm        = Lk.(fn{1});
    fn = fieldnames(Lv);  stride_kinematics_arm_vector = Lv.(fn{1});
    clear Lk Lv
    fprintf('[ok] loaded\n');
end

% upper_body_model.m sits next to this script in the repository.
this_dir = fileparts(mfilename('fullpath'));
if isempty(this_dir), this_dir = pwd; end
src = fileread(fullfile(this_dir, 'upper_body_model.m'));

% A unique ASCII code line is used as the injection anchor.
anchor = 'if APPLY_OFFSET_TO_FORWARD';
assert(count(src, anchor) == 1, 'anchor is not unique.');

% Optional overrides to reproduce a specific generation setting
if exist('FORCE_MODE','var') && ~isempty(FORCE_MODE)
    src = regexprep(src, "OFFSET_MODE\s*=\s*'[a-z]+';", ...
                    sprintf("OFFSET_MODE = '%s';", FORCE_MODE), 'once');
    fprintf('[patch] OFFSET_MODE -> %s\n', FORCE_MODE);
end
if exist('FORCE_ES_SCALE','var') && ~isempty(FORCE_ES_SCALE)
    src = regexprep(src, 'ES_SCALE\s*=\s*[-\d.]+;', ...
                    sprintf('ES_SCALE = %.6f;', FORCE_ES_SCALE), 'once');
    fprintf('[patch] ES_SCALE -> %.4f\n', FORCE_ES_SCALE);
end
if exist('FORCE_SC_SCALE','var') && ~isempty(FORCE_SC_SCALE)
    src = regexprep(src, 'SC_SCALE\s*=\s*[-\d.]+;', ...
                    sprintf('SC_SCALE = %.6f;', FORCE_SC_SCALE), 'once');
    fprintf('[patch] SC_SCALE -> %.4f\n', FORCE_SC_SCALE);
end

% Only ADDITIVE perturbation is used:  off_i <- off_i + deg2rad(DELTA)
%   Marker-placement bias acts as a constant identical for every stride, so
%   reproducing the observed deviation (~15 deg) literally calls for an
%   additive term, not proportional scaling.
IND = repmat(' ', 1, 16);
op_we = 'off_we = off_we + deg2rad(OFFSET_DELTA);';
op_es = 'off_es = off_es + deg2rad(OFFSET_DELTA);';
op_sc = 'off_sc = off_sc + deg2rad(OFFSET_DELTA);';
switch lower(SEG)
    case 'none',  lines = {};                                    % baseline
    case 'we',    lines = {op_we};
    case 'es',    lines = {op_es};
    case 'sc',    lines = {op_sc};
    case 'all',   lines = {op_we, op_es, op_sc};
    otherwise, error('unknown SEG=%s.', SEG);
end
inject = '';
for i = 1:numel(lines)
    inject = [inject lines{i} newline IND];  %#ok<AGROW>
end

if ~exist('DELTA_LIST','var'), DELTA_LIST = [-30, -15, 15, 30]; end

for p = DELTA_LIST
    fprintf('\n-------- SEG=%s  DELTA=%+.1f deg --------\n', SEG, p);
    OFFSET_DELTA = p;                                            %#ok<NASGU>
    patched = strrep(src, anchor, [inject anchor]);

    % NOTE: a '-' in the file name would make run() parse it as an expression,
    % so the sign is spelled out as a letter.
    if p < 0, tsgn = 'm'; else, tsgn = 'p'; end
    tmp = fullfile(tempdir, sprintf('ubm_%s_add_%s%d.m', ...
          lower(SEG), tsgn, abs(round(p*100))));
    fid = fopen(tmp,'w');  fwrite(fid, patched);  fclose(fid);

    old = cd(tempdir);
    try
        run(tmp);
    catch ME
        cd(old);  rethrow(ME);
    end
    cd(old);

    if strcmpi(SEG,'none')
        out = fullfile(OUT_DIR, 'stride_modeling_baseSC.mat');
    else
        if p < 0, sgn = 'm'; else, sgn = 'p'; end                % m=minus, p=plus
        out = fullfile(OUT_DIR, sprintf('stride_modeling_%sadd_%s%02d.mat', ...
              lower(SEG), sgn, abs(round(p))));
    end
    save(out, 'stride_modeling', '-v7.3');
    fprintf('[save] %s\n', out);
    clear stride_modeling
end

fprintf('\n=== done ===\n');
