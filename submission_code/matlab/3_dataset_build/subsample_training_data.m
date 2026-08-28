%% =====================================================================
%  Section 1) 피험자 수만 줄이는 small-data 생성 (S011 제외)
%  - N_subject_use 명을 stride_grf 기준으로 랜덤 선택
%  - [수정] S011은 ss3 데이터 누락 문제로 선택 후보에서 **무조건 제외**
%  - S001~S010, S011~S019 두 실험 셋팅에서 균형 있게 뽑기
%  - 선택된 Sxxx 목록은 다시 정렬해서 사용
% =======================================================================
clear; clc;

% --------- 설정 ---------
N_subject_use = 7;   % 사용할 피험자 수
% 재현 가능한 랜덤 샘플링을 원하면 seed 고정
rng(1);

% --------- 원본 병합 데이터 로드 ---------
load('merged_stride_grf_gyro.mat',                       'stride_grf');
load('merged_stride_kinematics_arm_gyro_offset_added.mat','stride_kinematics_arm');
load('merged_stride_kinematics_arm_vector_gyro.mat',     'stride_kinematics_arm_vector');
load('merged_stride_modeling.mat',                       'stride_modeling');

% --------- stride_grf 기준 Sxxx 피험자 목록 생성 ---------
subj_all = fieldnames(stride_grf);
maskS    = startsWith(subj_all,'S');
subj_all = subj_all(maskS);
subj_all = sort(subj_all);

% =========================================================
% [수정] S011 강제 제외 로직
% =========================================================
if ismember('S011', subj_all)
    subj_all = subj_all(~strcmp(subj_all, 'S011'));
    fprintf('[알림] "S011"은 데이터 불완전(ss3 부재)으로 인해 후보 목록에서 제외되었습니다.\n');
end
% =========================================================

N_total  = numel(subj_all);
if N_total == 0
    error('선택 가능한 피험자가 없습니다.');
end

% S번호를 숫자로 파싱해서 두 실험 셋팅 그룹 분리
subj_num = nan(N_total,1);
for i = 1:N_total
    tmp = sscanf(subj_all{i}, 'S%d');
    if ~isempty(tmp)
        subj_num(i) = tmp;
    end
end

% 두 그룹으로 분류 (S011은 이미 위에서 빠졌으므로 여기 포함 안 됨)
idx_grp1 = find(subj_num >= 1  & subj_num <= 10);
idx_grp2 = find(subj_num >  10);

subj_grp1 = subj_all(idx_grp1);
subj_grp2 = subj_all(idx_grp2);

N1 = numel(subj_grp1);   % 첫 번째 셋팅(S001~S010)
N2 = numel(subj_grp2);   % 두 번째 셋팅(S012~) - S011 제외됨

% --------- 사용할 피험자 선택 로직 ---------
if isempty(N_subject_use) || N_subject_use <= 0
    % 0 이하 혹은 비었으면 전체 사용
    subj_use = subj_all;
elseif N_subject_use >= N_total
    % 전체보다 많이 요청하면 전체 사용
    subj_use = subj_all;
else
    % 일부만 선택
    if N_subject_use == 1 || isempty(subj_grp1) || isempty(subj_grp2)
        % 한 명만 뽑거나, 한 그룹이 없으면 전체에서 랜덤
        idx_perm = randperm(N_total, N_subject_use);
        subj_use = subj_all(idx_perm);
    else
        % 균형 있게 뽑기
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
        
        % 부족분 채우기
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

% 실제 사용된 피험자 수
N_used    = numel(subj_use);
suffix_sub = sprintf('sub%02d', N_used);

fprintf('Section 1) 총 %d명(S011제외) 중 %d명 선택됨: %s ...\n', ...
    N_total, N_used, strjoin(subj_use', ', '));

% --------- 1-1) stride_grf 필터링 ---------
small_stride_grf = stride_grf;
fields_grf = fieldnames(stride_grf);
for iF = 1:numel(fields_grf)
    fn = fields_grf{iF};
    if startsWith(fn,'S') && ~ismember(fn, subj_use)
        small_stride_grf = rmfield(small_stride_grf, fn);
    end
end

% --------- 1-2) stride_modeling 필터링 ---------
small_stride_modeling = stride_modeling;
fields_mod = fieldnames(stride_modeling);
for iF = 1:numel(fields_mod)
    fn = fields_mod{iF};
    if startsWith(fn,'S') && ~ismember(fn, subj_use)
        small_stride_modeling = rmfield(small_stride_modeling, fn);
    end
end

% --------- 1-3) stride_kinematics_arm 필터링 ---------
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

% --------- 1-4) stride_kinematics_arm_vector 필터링 ---------
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

% --------- 1-5) 저장 ---------
stride_grf                    = small_stride_grf;
stride_modeling               = small_stride_modeling;
stride_kinematics_arm         = small_stride_kinematics_arm;
stride_kinematics_arm_vector  = small_stride_kinematics_arm_vector;

save(['merged_stride_grf_'                     suffix_sub '.mat'], 'stride_grf',                    '-v7.3');
save(['merged_stride_kinematics_arm_'          suffix_sub '.mat'], 'stride_kinematics_arm',         '-v7.3');
save(['merged_stride_kinematics_arm_vector_'   suffix_sub '.mat'], 'stride_kinematics_arm_vector',  '-v7.3');
save(['merged_stride_modeling_'                suffix_sub '.mat'], 'stride_modeling',               '-v7.3');

fprintf('Section 1) 저장 완료: *_%s.mat\n\n', suffix_sub);


%% =====================================================================
%  Section 2) stride(=cell) 개수만 줄이는 small-data 생성 (피험자 수는 그대로)
%  - 모든 Sxxx 피험자를 다 쓰고
%  - 각 Subject/Session/Day 에서 stride 개수를 25%, 50%, 75%로 줄인 버전 각각 저장
%  - 기준: stride_grf.(S).(SS).(DY).Left.Fx 의 cell 개수를 기준으로 잘라서
%          다른 모든 구조체의 cell들도 동일한 stride index만 남김 (1 ~ nKeep)
%  - 결과 파일 예:
%      merged_stride_grf_025.mat, merged_stride_kinematics_arm_025.mat, ...
%      merged_stride_grf_050.mat, ...
%      merged_stride_grf_075.mat, ...
% =======================================================================
clear; clc;

% --------- 설정 ---------
fractions = 0.05;   %

% --------- 원본 병합 데이터 로드 ---------
load('merged_stride_grf_gyro.mat',                 'stride_grf');
load('merged_stride_kinematics_arm_gyro_offset_added.mat', 'stride_kinematics_arm');
load('merged_stride_kinematics_arm_vector_gyro.mat',       'stride_kinematics_arm_vector');
load('merged_stride_modeling.mat',                'stride_modeling');

% --------- Sxxx 피험자 목록 (전체 사용) ---------
subj_all = fieldnames(stride_grf);
maskS    = startsWith(subj_all,'S');
subj_all = subj_all(maskS);
subj_all = sort(subj_all);
N_total  = numel(subj_all);

if N_total == 0
    error('stride_grf 안에 Sxxx 피험자가 없습니다.');
end

fprintf('Section 2) stride 개수 줄이기 대상 피험자 수: %d명\n', N_total);
disp(subj_all');

% --------- fraction(25/50/75%) 별로 small-data 생성 ---------
for iFrac = 1:numel(fractions)
    frac = fractions(iFrac);
    fracTag = sprintf('%03d', round(frac*100));  % 0.25 -> '025' 등

    fprintf('\n=== FRAC_KEEP = %.2f (suffix = %s) 처리 시작 ===\n', frac, fracTag);

    % 원본 복사 후 이 복사본에서만 stride cell 개수 줄임
    sg  = stride_grf;
    ska = stride_kinematics_arm;
    skv = stride_kinematics_arm_vector;
    sm  = stride_modeling;

    % --------- Subject / Session / Day 루프 ---------
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

                % ---- 이 Trial 의 stride 개수 확인 (Left.Fx 기준) ----
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

                % ---- 남길 stride 개수 계산 (앞에서부터 연속) ----
                nKeep = floor(frac * nStride);
                if nKeep < 1
                    nKeep = 1;
                elseif nKeep > nStride
                    nKeep = nStride;
                end
                keepIdx = 1:nKeep;  % 마지막으로 남기는 stride 인덱스 = nKeep

                % --------------------------------------------------
                % 2-1) stride_grf: Left/Right/Total 의 모든 cell 필드 잘라주기
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
                % 2-2) stride_kinematics_arm: segment 밑의 모든 cell 필드 잘라주기
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
                % 2-3) stride_kinematics_arm_vector: pair 밑 구조/셀 모두 잘라주기
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
                            % 1단계: 바로 cell array 인 경우
                            nodeV.(fld1) = val1(keepIdx);

                        elseif isstruct(val1)
                            % 2단계: 하위 struct 안에 cell array 들이 있는 경우
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
                % 2-4) stride_modeling: S/SS/Day 밑의 cell 필드 잘라주기
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

    % --------- 잘린 복사본을 원래 변수 이름에 넣고 저장 ---------
    stride_grf              = sg;
    stride_kinematics_arm   = ska;
    stride_kinematics_arm_vector = skv;
    stride_modeling         = sm;

    save(['merged_stride_grf_'            fracTag '.mat'], 'stride_grf',              '-v7.3');
    save(['merged_stride_kinematics_arm_' fracTag '.mat'], 'stride_kinematics_arm',   '-v7.3');
    save(['merged_stride_kinematics_arm_vector_' fracTag '.mat'], 'stride_kinematics_arm_vector','-v7.3');
    save(['merged_stride_modeling_'       fracTag '.mat'], 'stride_modeling',         '-v7.3');

    fprintf('  -> FRAC_KEEP=%.2f 완료: *_%s.mat 저장\n', frac, fracTag);
end

