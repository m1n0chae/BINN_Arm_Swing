%% =====================================================================
%  Section 1) Small-data variant that reduces the NUMBER OF SUBJECTS (S011 excluded)
%  - randomly selects N_subject_use subjects based on stride_grf
%  - S011 is ALWAYS excluded from the candidate pool (its ss3 session is missing)
%  - subjects are drawn in a balanced way from the two experiment settings
%    (S001-S010 and S012-S019)
%  - the selected Sxxx list is re-sorted before use
%  NOTE: this section is auxiliary; the training-set-size results in the paper
%        use the per-stride subsampling of Section 2, not this section.
% =======================================================================
clear; clc;

% --------- settings ---------
N_subject_use = 7;   % number of subjects to keep
% fix the seed for reproducible random sampling
rng(1);

% --------- load the merged source data ---------
load('merged_stride_grf_gyro.mat',                       'stride_grf');
load('merged_stride_kinematics_arm_gyro_offset_added.mat','stride_kinematics_arm');
load('merged_stride_kinematics_arm_vector_gyro.mat',     'stride_kinematics_arm_vector');
load('merged_stride_modeling.mat',                       'stride_modeling');

% --------- Sxxx subject list based on stride_grf ---------
subj_all = fieldnames(stride_grf);
maskS    = startsWith(subj_all,'S');
subj_all = subj_all(maskS);
subj_all = sort(subj_all);

% =========================================================
% force-exclude S011
% =========================================================
if ismember('S011', subj_all)
    subj_all = subj_all(~strcmp(subj_all, 'S011'));
    fprintf('[note] "S011" excluded from the candidate list (incomplete data: ss3 missing).\n');
end
% =========================================================

N_total  = numel(subj_all);
if N_total == 0
    error('No selectable subjects.');
end

% parse the S numbers to split the two experiment-setting groups
subj_num = nan(N_total,1);
for i = 1:N_total
    tmp = sscanf(subj_all{i}, 'S%d');
    if ~isempty(tmp)
        subj_num(i) = tmp;
    end
end

% two groups (S011 is already gone, so it is not included here)
idx_grp1 = find(subj_num >= 1  & subj_num <= 10);
idx_grp2 = find(subj_num >  10);

subj_grp1 = subj_all(idx_grp1);
subj_grp2 = subj_all(idx_grp2);

N1 = numel(subj_grp1);   % first setting (S001-S010)
N2 = numel(subj_grp2);   % second setting (S012-)

% --------- subject-selection logic ---------
if isempty(N_subject_use) || N_subject_use <= 0
    % empty or <= 0: use everyone
    subj_use = subj_all;
elseif N_subject_use >= N_total
    % more than available: use everyone
    subj_use = subj_all;
else
    % select a subset
    if N_subject_use == 1 || isempty(subj_grp1) || isempty(subj_grp2)
        % one subject, or one group empty: draw from the whole pool
        idx_perm = randperm(N_total, N_subject_use);
        subj_use = subj_all(idx_perm);
    else
        % balanced draw
        target1 = ceil(N_subject_use / 2);
        target2 = N_subject_use - target1;

        target1 = min(target1, N1);
        target2 = min(target2, N2);

        sel1 = {}; sel2 = {};
        if target1 > 0 && N1 > 0
            perm1 = randperm(N1, target1);
            sel1 = subj_grp1(perm1);
        end
        if target2 > 0 && N2 > 0
            perm2 = randperm(N2, target2);
            sel2 = subj_grp2(perm2);
        end

        subj_tmp = [sel1; sel2];

        % fill any shortfall
        N_current = numel(subj_tmp);
        if N_current < N_subject_use
            mask_used   = ismember(subj_all, subj_tmp);
            remain_all  = subj_all(~mask_used);
            N_remain    = numel(remain_all);
            N_needed    = min(N_subject_use - N_current, N_remain);

            if N_needed > 0
                perm_extra = randperm(N_remain, N_needed);
                extra      = remain_all(perm_extra);
                subj_tmp   = [subj_tmp; extra];
            end
        end

        subj_use = sort(subj_tmp);
    end
end

% actual number of subjects used
N_used    = numel(subj_use);
suffix_sub = sprintf('sub%02d', N_used);

fprintf('Section 1) selected %d of %d subjects (S011 excluded): %s ...\n', ...
    N_used, N_total, strjoin(subj_use', ', '));

% --------- 1-1) filter stride_grf ---------
small_stride_grf = stride_grf;
fields_grf = fieldnames(stride_grf);
for iF = 1:numel(fields_grf)
    fn = fields_grf{iF};
    if startsWith(fn,'S') && ~ismember(fn, subj_use)
        small_stride_grf = rmfield(small_stride_grf, fn);
    end
end

% --------- 1-2) filter stride_modeling ---------
small_stride_modeling = stride_modeling;
fields_mod = fieldnames(stride_modeling);
for iF = 1:numel(fields_mod)
    fn = fields_mod{iF};
    if startsWith(fn,'S') && ~ismember(fn, subj_use)
        small_stride_modeling = rmfield(small_stride_modeling, fn);
    end
end

% --------- 1-3) filter stride_kinematics_arm ---------
small_stride_kinematics_arm = stride_kinematics_arm;
segments = fieldnames(stride_kinematics_arm);
for iSeg = 1:numel(segments)
    seg = segments{iSeg};
    if ~isstruct(stride_kinematics_arm.(seg)), continue; end

    sub_in_seg = fieldnames(stride_kinematics_arm.(seg));
    for iS = 1:numel(sub_in_seg)
        S = sub_in_seg{iS};
        if startsWith(S,'S') && ~ismember(S, subj_use)
            small_stride_kinematics_arm.(seg) = rmfield(small_stride_kinematics_arm.(seg), S);
        end
    end
end

% --------- 1-4) filter stride_kinematics_arm_vector ---------
small_stride_kinematics_arm_vector = stride_kinematics_arm_vector;
pairs = fieldnames(stride_kinematics_arm_vector);
for iP = 1:numel(pairs)
    pair = pairs{iP};
    if ~isstruct(stride_kinematics_arm_vector.(pair)), continue; end

    sub_in_pair = fieldnames(stride_kinematics_arm_vector.(pair));
    for iS = 1:numel(sub_in_pair)
        S = sub_in_pair{iS};
        if startsWith(S,'S') && ~ismember(S, subj_use)
            small_stride_kinematics_arm_vector.(pair) = rmfield(small_stride_kinematics_arm_vector.(pair), S);
        end
    end
end

% --------- 1-5) save ---------
stride_grf                    = small_stride_grf;
stride_modeling               = small_stride_modeling;
stride_kinematics_arm         = small_stride_kinematics_arm;
stride_kinematics_arm_vector  = small_stride_kinematics_arm_vector;

save(['merged_stride_grf_'                     suffix_sub '.mat'], 'stride_grf',                    '-v7.3');
save(['merged_stride_kinematics_arm_'          suffix_sub '.mat'], 'stride_kinematics_arm',         '-v7.3');
save(['merged_stride_kinematics_arm_vector_'   suffix_sub '.mat'], 'stride_kinematics_arm_vector',  '-v7.3');
save(['merged_stride_modeling_'                suffix_sub '.mat'], 'stride_modeling',               '-v7.3');

fprintf('Section 1) saved: *_%s.mat\n\n', suffix_sub);


%% =====================================================================
%  Section 2) Small-data variants that reduce the NUMBER OF STRIDES
%             (random stride-wise subsampling; all subjects kept)
%  - keeps every Sxxx subject
%  - within each Subject/Session/Day trial, nKeep = round(frac * nStride)
%    strides are drawn AT RANDOM (without replacement) and kept in their
%    original chronological order
%  - repeated for several independent cases (random seeds), giving the
%    case1/case2/case3 variants averaged in the paper for fractions < 100%
%  - the stride count of stride_grf.(S).(SS).(DY).Left.Fx defines the trial
%    length; the same kept-stride indices are applied to every other struct
%  - output files, e.g.:
%      merged_stride_grf_005_day1_case1.mat, merged_stride_kinematics_arm_005_day1_case1.mat, ...
%
%  NOTE: this reproduces the sampling procedure of the released datasets
%  (verified against them: random, chronologically sorted, nKeep = round);
%  the RNG seeds of the original generation run were not recorded, so a
%  fresh run draws a different random subset. The released *_case*.mat
%  files are the datasets of record.
% =======================================================================
clear; clc;

% --------- settings ---------
fractions = [0.05 0.10 0.20 0.50];   % training-set fractions
cases     = 1:3;                     % independent random subsamplings

% --------- load the merged source data ---------
load('merged_stride_grf_gyro.mat',                 'stride_grf');
load('merged_stride_kinematics_arm_gyro_offset_added.mat', 'stride_kinematics_arm');
load('merged_stride_kinematics_arm_vector_gyro.mat',       'stride_kinematics_arm_vector');
load('merged_stride_modeling.mat',                'stride_modeling');

% --------- Sxxx subject list (everyone) ---------
subj_all = fieldnames(stride_grf);
maskS    = startsWith(subj_all,'S');
subj_all = subj_all(maskS);
subj_all = sort(subj_all);
N_total  = numel(subj_all);

if N_total == 0
    error('No Sxxx subjects inside stride_grf.');
end

fprintf('Section 2) subjects for stride subsampling: %d\n', N_total);
disp(subj_all');

% --------- generate the small-data variants per fraction and case ---------
for iFrac = 1:numel(fractions)
  frac = fractions(iFrac);
  for iCase = 1:numel(cases)
    caseNo  = cases(iCase);
    fracTag = sprintf('%03d_day1_case%d', round(frac*100), caseNo);  % 0.05, case1 -> '005_day1_case1'

    fprintf('\n=== FRAC_KEEP = %.2f  case %d (suffix = %s) ===\n', frac, caseNo, fracTag);

    % one deterministic seed per (fraction, case)
    rng(1000*round(frac*100) + caseNo);

    % copy the sources; strides are removed from the copies only
    sg  = stride_grf;
    ska = stride_kinematics_arm;
    skv = stride_kinematics_arm_vector;
    sm  = stride_modeling;

    % --------- Subject / Session / Day loop ---------
    for iS = 1:numel(subj_all)
        S = subj_all{iS};
        if ~isfield(sg, S), continue; end

        sessNames = fieldnames(sg.(S));
        for iSS = 1:numel(sessNames)
            SS = sessNames{iSS};
            if ~isstruct(sg.(S).(SS)), continue; end

            dayNames = fieldnames(sg.(S).(SS));
            for iD = 1:numel(dayNames)
                DY = dayNames{iD};
                if ~isstruct(sg.(S).(SS).(DY)), continue; end

                % ---- stride count of this trial (based on Left.Fx) ----
                if ~isfield(sg.(S).(SS).(DY),'Left') || ...
                   ~isfield(sg.(S).(SS).(DY).Left,'Fx')
                    continue;
                end

                L_Fx_cells = sg.(S).(SS).(DY).Left.Fx;
                if ~iscell(L_Fx_cells) || isempty(L_Fx_cells)
                    continue;
                end

                nStride = numel(L_Fx_cells);
                if nStride == 0
                    continue;
                end

                % ---- number of strides to keep and their random indices ----
                nKeep = round(frac * nStride);
                if nKeep < 1
                    nKeep = 1;
                elseif nKeep > nStride
                    nKeep = nStride;
                end
                keepIdx = sort(randperm(nStride, nKeep));  % random subset, chronological order

                % --------------------------------------------------
                % 2-1) stride_grf: slice every cell field of Left/Right/Total
                % --------------------------------------------------
                sideNames = {'Left','Right','Total'};
                for iSide = 1:numel(sideNames)
                    sd = sideNames{iSide};
                    if ~isfield(sg.(S).(SS).(DY), sd), continue; end

                    comps = fieldnames(sg.(S).(SS).(DY).(sd));
                    for ic = 1:numel(comps)
                        comp = comps{ic};
                        val  = sg.(S).(SS).(DY).(sd).(comp);
                        if iscell(val) && numel(val) >= nKeep
                            sg.(S).(SS).(DY).(sd).(comp) = val(keepIdx);
                        end
                    end
                end

                % --------------------------------------------------
                % 2-2) stride_kinematics_arm: slice every cell field under each segment
                % --------------------------------------------------
                segNames = fieldnames(ska);
                for iSeg = 1:numel(segNames)
                    seg = segNames{iSeg};
                    if ~isfield(ska.(seg), S) || ...
                       ~isfield(ska.(seg).(S), SS) || ...
                       ~isfield(ska.(seg).(S).(SS), DY)
                        continue;
                    end

                    nodeA = ska.(seg).(S).(SS).(DY);
                    fA = fieldnames(nodeA);
                    for ia = 1:numel(fA)
                        fld = fA{ia};
                        val = nodeA.(fld);
                        if iscell(val) && numel(val) >= nKeep
                            nodeA.(fld) = val(keepIdx);
                        end
                    end
                    ska.(seg).(S).(SS).(DY) = nodeA;
                end

                % --------------------------------------------------
                % 2-3) stride_kinematics_arm_vector: slice cells under each pair (nested too)
                % --------------------------------------------------
                pairNames = fieldnames(skv);
                for iP = 1:numel(pairNames)
                    pair = pairNames{iP};
                    if ~isfield(skv.(pair), S) || ...
                       ~isfield(skv.(pair).(S), SS) || ...
                       ~isfield(skv.(pair).(S).(SS), DY)
                        continue;
                    end

                    nodeV = skv.(pair).(S).(SS).(DY);
                    fV1 = fieldnames(nodeV);

                    for iv1 = 1:numel(fV1)
                        fld1 = fV1{iv1};
                        val1 = nodeV.(fld1);

                        if iscell(val1) && numel(val1) >= nKeep
                            % level 1: the field is a cell array itself
                            nodeV.(fld1) = val1(keepIdx);

                        elseif isstruct(val1)
                            % level 2: the field is a struct holding cell arrays
                            fV2 = fieldnames(val1);
                            for iv2 = 1:numel(fV2)
                                fld2 = fV2{iv2};
                                val2 = val1.(fld2);
                                if iscell(val2) && numel(val2) >= nKeep
                                    val1.(fld2) = val2(keepIdx);
                                end
                            end
                            nodeV.(fld1) = val1;
                        end
                    end

                    skv.(pair).(S).(SS).(DY) = nodeV;
                end

                % --------------------------------------------------
                % 2-4) stride_modeling: slice the cell fields under S/SS/Day
                % --------------------------------------------------
                if isfield(sm, S) && isfield(sm.(S), SS) && isfield(sm.(S).(SS), DY)
                    nodeM = sm.(S).(SS).(DY);
                    fM = fieldnames(nodeM);
                    for im = 1:numel(fM)
                        fldm = fM{im};
                        valm = nodeM.(fldm);
                        if iscell(valm) && numel(valm) >= nKeep
                            nodeM.(fldm) = valm(keepIdx);
                        end
                    end
                    sm.(S).(SS).(DY) = nodeM;
                end

            end  % Day
        end  % Session
    end  % Subject

    % --------- save the sliced copies under the canonical variable names ---------
    % (the workspace sources are swapped out and restored so the next
    %  fraction/case starts from the full data again)
    tmp_grf = stride_grf;  tmp_ska = stride_kinematics_arm;
    tmp_skv = stride_kinematics_arm_vector;  tmp_sm = stride_modeling;

    stride_grf = sg;  stride_kinematics_arm = ska;
    stride_kinematics_arm_vector = skv;  stride_modeling = sm;

    save(['merged_stride_grf_'            fracTag '.mat'], 'stride_grf',              '-v7.3');
    save(['merged_stride_kinematics_arm_' fracTag '.mat'], 'stride_kinematics_arm',   '-v7.3');
    save(['merged_stride_kinematics_arm_vector_' fracTag '.mat'], 'stride_kinematics_arm_vector','-v7.3');
    save(['merged_stride_modeling_'       fracTag '.mat'], 'stride_modeling',         '-v7.3');

    stride_grf = tmp_grf;  stride_kinematics_arm = tmp_ska;
    stride_kinematics_arm_vector = tmp_skv;  stride_modeling = tmp_sm;
    clear tmp_grf tmp_ska tmp_skv tmp_sm

    fprintf('  -> FRAC_KEEP=%.2f case %d done: *_%s.mat saved\n', frac, caseNo, fracTag);
  end
end
