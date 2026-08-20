%% ===================== Speed-wise P2P & Offset 분석 =====================
% 전제: stride_kinematics_arm_vector 변수가 workspace에 존재해야 함.
%   구조 예시:
%   stride_kinematics_arm_vector.(REL).(Subject).(Session).(Day).time.sagittal_angle.pos{k}
%   REL: 'L_Wrist_to_L_Elbow', 'L_Elbow_to_L_Shoulder', 'L_Shoulder_to_sacrum'
%   Subject: 'S001' 등
%   Session: 'ss1','ss2','ss3'
%   Day: 'Day1','Day2', ...

clc;
clearvars -except stride_kinematics_arm_vector

if ~exist('stride_kinematics_arm_vector','var')
    error('stride_kinematics_arm_vector 변수가 없습니다. 먼저 stride_kinematics_arm_vector_gyro.mat를 로드하세요.');
end

%% 0) REL / Subject / Session / Speed 매핑 설정

REL_WE = 'L_Wrist_to_L_Elbow';       % WE
REL_ES = 'L_Elbow_to_L_Shoulder';    % ES
REL_SC = 'L_Shoulder_to_sacrum'; % SC
REL_LIST = {REL_WE, REL_ES, REL_SC};

% Subject / Session을 추출할 기준 REL
if isfield(stride_kinematics_arm_vector, REL_WE)
    rel_for_list = REL_WE;
elseif isfield(stride_kinematics_arm_vector, REL_ES)
    rel_for_list = REL_ES;
elseif isfield(stride_kinematics_arm_vector, REL_SC)
    rel_for_list = REL_SC;
else
    error('stride_kinematics_arm_vector에 WE/ES/SC REL이 없습니다.');
end

% Subject 목록: S*** 필드만 사용
subs_all  = fieldnames(stride_kinematics_arm_vector.(rel_for_list));
subs_mask = startsWith(subs_all, 'S');
subjects  = subs_all(subs_mask);
subjects  = subjects(:)';

if isempty(subjects)
    error('%s 안에서 S*** subject 필드를 찾지 못했습니다.', rel_for_list);
end

% Session 목록: ss* 필드만 사용 (ss1~ss3)
sess_all  = fieldnames(stride_kinematics_arm_vector.(rel_for_list).(subjects{1}));
sess_mask = startsWith(sess_all, 'ss');
sessions  = sess_all(sess_mask);
sessions  = sessions(:)';

fprintf('=== 분석 대상 ===\n');
fprintf('- Subjects: %s ... %s (총 %d명)\n', subjects{1}, subjects{end}, numel(subjects));
fprintf('- Sessions: %s (총 %d개)\n\n', strjoin(sessions', ', '), numel(sessions));

% 속도 매핑 (사용자 실험에서 ss1~ss3만 있다고 하셨으므로)
speed_map = table({'ss1';'ss2';'ss3'}, ...
                  [1.00; 1.25; 1.50], ...
                  'VariableNames', {'Session','Speed'});
speeds    = speed_map.Speed(:)';
sess_list = speed_map.Session;

%% 1) stride별 P2P / Offset(Mean) 수집 (WE / ES / SC)

rows_WE_p2p = {};  % {Subject, Session, Day, CycleIdx, WE_P2P}
rows_ES_p2p = {};
rows_SC_p2p = {};

rows_WE_off = {};  % {Subject, Session, Day, CycleIdx, WE_Mean}
rows_ES_off = {};
rows_SC_off = {};

for r = 1:numel(REL_LIST)
    REL = REL_LIST{r};
    if ~isfield(stride_kinematics_arm_vector, REL)
        fprintf('[SKIP] %s: stride_kinematics_arm_vector에 없음\n', REL);
        continue;
    end

    for iS = 1:numel(subjects)
        S = subjects{iS};
        if ~isfield(stride_kinematics_arm_vector.(REL), S)
            continue;
        end

        for iSS = 1:numel(sessions)
            SS = sessions{iSS};
            if ~isfield(stride_kinematics_arm_vector.(REL).(S), SS)
                continue;
            end

            % Day 목록 (Day1, Day2, ...) - 없으면 Day 없이 SS 아래에 바로 stride가 있을 수도 있음
            day_all  = fieldnames(stride_kinematics_arm_vector.(REL).(S).(SS));
            day_mask = startsWith(day_all, 'Day');
            days     = day_all(day_mask);
            if isempty(days)
                days = {''};
            end

            for iD = 1:numel(days)
                DY = days{iD};

                if isempty(DY)
                    node = stride_kinematics_arm_vector.(REL).(S).(SS);
                else
                    if ~isfield(stride_kinematics_arm_vector.(REL).(S).(SS), DY)
                        continue;
                    end
                    node = stride_kinematics_arm_vector.(REL).(S).(SS).(DY);
                end

                % --- stride angle cell 찾기 ---
                segCells = [];

                % 1) node.time.sagittal_angle.pos (cell array) 체크
                if isfield(node,'time') && isstruct(node.time) && ...
                   isfield(node.time,'sagittal_angle') && ...
                   isfield(node.time.sagittal_angle,'pos') && ...
                   iscell(node.time.sagittal_angle.pos)
                    segCells = node.time.sagittal_angle.pos;
                end

                % 2) node.sagittal_angle.pos (cell array) 형태도 지원
                if isempty(segCells)
                    if isfield(node,'sagittal_angle') && isstruct(node.sagittal_angle) && ...
                       isfield(node.sagittal_angle,'pos') && ...
                       iscell(node.sagittal_angle.pos)
                        segCells = node.sagittal_angle.pos;
                    end
                end

                if isempty(segCells)
                    fprintf('[SKIP] %s-%s-%s-%s: stride sagittal_angle.pos cell 없음\n', REL, S, SS, DY);
                    continue;
                end

                nStride = numel(segCells);
                if nStride == 0
                    continue;
                end

                % --- 각 stride의 P2P 및 Mean 계산 ---
                for kk = 1:nStride
                    y = segCells{kk};
                    if isempty(y) || ~isnumeric(y)
                        continue;
                    end
                    y = double(y(:));

                    p2p_val  = max(y) - min(y);
                    mean_val = mean(y,'omitnan');

                    if isfinite(p2p_val)
                        switch REL
                            case REL_WE
                                rows_WE_p2p(end+1,:) = {S, SS, DY, kk, p2p_val}; %#ok<SAGROW>
                            case REL_ES
                                rows_ES_p2p(end+1,:) = {S, SS, DY, kk, p2p_val}; %#ok<SAGROW>
                            case REL_SC
                                rows_SC_p2p(end+1,:) = {S, SS, DY, kk, p2p_val}; %#ok<SAGROW>
                        end
                    end

                    if isfinite(mean_val)
                        switch REL
                            case REL_WE
                                rows_WE_off(end+1,:) = {S, SS, DY, kk, mean_val}; %#ok<SAGROW>
                            case REL_ES
                                rows_ES_off(end+1,:) = {S, SS, DY, kk, mean_val}; %#ok<SAGROW>
                            case REL_SC
                                rows_SC_off(end+1,:) = {S, SS, DY, kk, mean_val}; %#ok<SAGROW>
                        end
                    end
                end
            end
        end
    end
end

% 빈 경우에도 cell2table 에러 방지
if isempty(rows_WE_p2p), rows_WE_p2p = cell(0,5); end
if isempty(rows_ES_p2p), rows_ES_p2p = cell(0,5); end
if isempty(rows_SC_p2p), rows_SC_p2p = cell(0,5); end
if isempty(rows_WE_off), rows_WE_off = cell(0,5); end
if isempty(rows_ES_off), rows_ES_off = cell(0,5); end
if isempty(rows_SC_off), rows_SC_off = cell(0,5); end

T_WE_p2p = cell2table(rows_WE_p2p,'VariableNames',{'Subject','Session','Day','CycleIdx','WE_P2P'});
T_ES_p2p = cell2table(rows_ES_p2p,'VariableNames',{'Subject','Session','Day','CycleIdx','ES_P2P'});
T_SC_p2p = cell2table(rows_SC_p2p,'VariableNames',{'Subject','Session','Day','CycleIdx','SC_P2P'});

T_WE_off = cell2table(rows_WE_off,'VariableNames',{'Subject','Session','Day','CycleIdx','WE_Mean'});
T_ES_off = cell2table(rows_ES_off,'VariableNames',{'Subject','Session','Day','CycleIdx','ES_Mean'});
T_SC_off = cell2table(rows_SC_off,'VariableNames',{'Subject','Session','Day','CycleIdx','SC_Mean'});

% 문자열 형식 맞추기
T_WE_p2p.Subject = cellstr(string(T_WE_p2p.Subject));
T_ES_p2p.Subject = cellstr(string(T_ES_p2p.Subject));
T_SC_p2p.Subject = cellstr(string(T_SC_p2p.Subject));
T_WE_p2p.Session = cellstr(string(T_WE_p2p.Session));
T_ES_p2p.Session = cellstr(string(T_ES_p2p.Session));
T_SC_p2p.Session = cellstr(string(T_SC_p2p.Session));
T_WE_p2p.Day     = cellstr(string(T_WE_p2p.Day));
T_ES_p2p.Day     = cellstr(string(T_ES_p2p.Day));
T_SC_p2p.Day     = cellstr(string(T_SC_p2p.Day));

T_WE_off.Subject = cellstr(string(T_WE_off.Subject));
T_ES_off.Subject = cellstr(string(T_ES_off.Subject));
T_SC_off.Subject = cellstr(string(T_SC_off.Subject));
T_WE_off.Session = cellstr(string(T_WE_off.Session));
T_ES_off.Session = cellstr(string(T_ES_off.Session));
T_SC_off.Session = cellstr(string(T_SC_off.Session));
T_WE_off.Day     = cellstr(string(T_WE_off.Day));
T_ES_off.Day     = cellstr(string(T_ES_off.Day));
T_SC_off.Day     = cellstr(string(T_SC_off.Day));

disp('=== stride 기반 P2P / Offset 값 수집 완료 ===');


%% 2) P2P 기준: 속도(speed-wise) 분석 (WE / ES / SC + 비율)
clc;

%% 2-1) WE / ES / SC 각자 speed-wise P2P mean±SD

% WE
if ~isempty(T_WE_p2p)
    [Gsess_we, SessList_we] = findgroups(T_WE_p2p.Session);
    we_mean_s = splitapply(@mean, T_WE_p2p.WE_P2P, Gsess_we);
    we_std_s  = splitapply(@std,  T_WE_p2p.WE_P2P, Gsess_we);
    T_WE_bySess_p2p = table(SessList_we, we_mean_s, we_std_s, ...
                            'VariableNames',{'Session','MeanP2P','StdP2P'});
    T_WE_bySess_p2p = innerjoin(T_WE_bySess_p2p, speed_map, 'Keys','Session');
else
    T_WE_bySess_p2p = table(cell(0,1), [], [], [], ...
                            'VariableNames',{'Session','MeanP2P','StdP2P','Speed'});
end

% ES
if ~isempty(T_ES_p2p)
    [Gsess_es, SessList_es] = findgroups(T_ES_p2p.Session);
    es_mean_s = splitapply(@mean, T_ES_p2p.ES_P2P, Gsess_es);
    es_std_s  = splitapply(@std,  T_ES_p2p.ES_P2P, Gsess_es);
    T_ES_bySess_p2p = table(SessList_es, es_mean_s, es_std_s, ...
                            'VariableNames',{'Session','MeanP2P','StdP2P'});
    T_ES_bySess_p2p = innerjoin(T_ES_bySess_p2p, speed_map, 'Keys','Session');
else
    T_ES_bySess_p2p = table(cell(0,1), [], [], [], ...
                            'VariableNames',{'Session','MeanP2P','StdP2P','Speed'});
end

% SC
if ~isempty(T_SC_p2p)
    [Gsess_sc, SessList_sc] = findgroups(T_SC_p2p.Session);
    sc_mean_s = splitapply(@mean, T_SC_p2p.SC_P2P, Gsess_sc);
    sc_std_s  = splitapply(@std,  T_SC_p2p.SC_P2P, Gsess_sc);
    T_SC_bySess_p2p = table(SessList_sc, sc_mean_s, sc_std_s, ...
                            'VariableNames',{'Session','MeanP2P','StdP2P'});
    T_SC_bySess_p2p = innerjoin(T_SC_bySess_p2p, speed_map, 'Keys','Session');
else
    T_SC_bySess_p2p = table(cell(0,1), [], [], [], ...
                            'VariableNames',{'Session','MeanP2P','StdP2P','Speed'});
end

[sp_we,ord_we_s] = sort(T_WE_bySess_p2p.Speed);
[sp_es,ord_es_s] = sort(T_ES_bySess_p2p.Speed);
[sp_sc,ord_sc_s] = sort(T_SC_bySess_p2p.Speed);

figure('Color','w','Position',[100 100 1200 800], ...
       'Name','P2P Speed-wise mean±SD (WE / ES / SC)','NumberTitle','off');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(1); hold(ax1,'on'); grid(ax1,'on'); box(ax1,'on');
if ~isempty(sp_we)
    errorbar(ax1, sp_we, T_WE_bySess_p2p.MeanP2P(ord_we_s), T_WE_bySess_p2p.StdP2P(ord_we_s), '-o','LineWidth',1.8);
end
xlabel(ax1,'Speed (m/s)'); ylabel(ax1,'WE P2P');
title(ax1,'WE: Speed-wise P2P');

ax2 = nexttile(2); hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on');
if ~isempty(sp_es)
    errorbar(ax2, sp_es, T_ES_bySess_p2p.MeanP2P(ord_es_s), T_ES_bySess_p2p.StdP2P(ord_es_s), '-o','LineWidth',1.8);
end
xlabel(ax2,'Speed (m/s)'); ylabel(ax2,'ES P2P');
title(ax2,'ES: Speed-wise P2P');

ax3 = nexttile(3); hold(ax3,'on'); grid(ax3,'on'); box(ax3,'on');
if ~isempty(sp_sc)
    errorbar(ax3, sp_sc, T_SC_bySess_p2p.MeanP2P(ord_sc_s), T_SC_bySess_p2p.StdP2P(ord_sc_s), '-o','LineWidth',1.8);
end
xlabel(ax3,'Speed (m/s)'); ylabel(ax3,'SC P2P');
title(ax3,'SC: Speed-wise P2P');

%% 2-2) P2P 비율: ES/WE, SC/WE, WE/SC (Speed-wise)

clc;

% --- trial 단위(Subject×Session×Day) mean P2P 구하기 ---
% ES/WE
T_ratio_ES_WE_p2p = table();
if ~isempty(T_ES_p2p) && ~isempty(T_WE_p2p)
    [Gtrial_es, Sub_es, Sess_es, Day_es] = findgroups(T_ES_p2p.Subject, T_ES_p2p.Session, T_ES_p2p.Day);
    es_trial_mean = splitapply(@mean, T_ES_p2p.ES_P2P, Gtrial_es);
    T_ES_trial_p2p = table(Sub_es, Sess_es, Day_es, es_trial_mean, ...
                           'VariableNames',{'Subject','Session','Day','ES_MeanP2P'});

    [Gtrial_we, Sub_we_t, Sess_we_t, Day_we] = findgroups(T_WE_p2p.Subject, T_WE_p2p.Session, T_WE_p2p.Day);
    we_trial_mean = splitapply(@mean, T_WE_p2p.WE_P2P, Gtrial_we);
    T_WE_trial_p2p = table(Sub_we_t, Sess_we_t, Day_we, we_trial_mean, ...
                           'VariableNames',{'Subject','Session','Day','WE_MeanP2P'});

    T_ratio_ES_WE_p2p = innerjoin(T_ES_trial_p2p, T_WE_trial_p2p, 'Keys',{'Subject','Session','Day'});
    T_ratio_ES_WE_p2p.Ratio = T_ratio_ES_WE_p2p.ES_MeanP2P ./ T_ratio_ES_WE_p2p.WE_MeanP2P;
    T_ratio_ES_WE_p2p = innerjoin(T_ratio_ES_WE_p2p, speed_map, 'Keys','Session');
end

% SC/WE, WE/SC
T_ratio_SC_WE_p2p = table();
T_ratio_WE_SC_p2p = table();
if ~isempty(T_SC_p2p) && ~isempty(T_WE_p2p)
    [Gtrial_sc, Sub_sc, Sess_sc, Day_sc] = findgroups(T_SC_p2p.Subject, T_SC_p2p.Session, T_SC_p2p.Day);
    sc_trial_mean = splitapply(@mean, T_SC_p2p.SC_P2P, Gtrial_sc);
    T_SC_trial_p2p = table(Sub_sc, Sess_sc, Day_sc, sc_trial_mean, ...
                           'VariableNames',{'Subject','Session','Day','SC_MeanP2P'});

    [Gtrial_we2, Sub_we2_t, Sess_we2_t, Day_we2] = findgroups(T_WE_p2p.Subject, T_WE_p2p.Session, T_WE_p2p.Day);
    we_trial_mean2 = splitapply(@mean, T_WE_p2p.WE_P2P, Gtrial_we2);
    T_WE2_trial_p2p = table(Sub_we2_t, Sess_we2_t, Day_we2, we_trial_mean2, ...
                            'VariableNames',{'Subject','Session','Day','WE_MeanP2P'});

    T_ratio_SC_WE_p2p = innerjoin(T_SC_trial_p2p, T_WE2_trial_p2p, 'Keys',{'Subject','Session','Day'});
    T_ratio_SC_WE_p2p.Ratio = T_ratio_SC_WE_p2p.SC_MeanP2P ./ T_ratio_SC_WE_p2p.WE_MeanP2P;
    T_ratio_SC_WE_p2p = innerjoin(T_ratio_SC_WE_p2p, speed_map, 'Keys','Session');

    % 반대로 WE/SC도 계산
    T_ratio_WE_SC_p2p = innerjoin(T_WE2_trial_p2p, T_SC_trial_p2p, 'Keys',{'Subject','Session','Day'});
    T_ratio_WE_SC_p2p.Ratio = T_ratio_WE_SC_p2p.WE_MeanP2P ./ T_ratio_WE_SC_p2p.SC_MeanP2P;
    T_ratio_WE_SC_p2p = innerjoin(T_ratio_WE_SC_p2p, speed_map, 'Keys','Session');
end

% --- speed-wise로 집계 (Session 기준 평균±SD) ---

% ES/WE
if ~isempty(T_ratio_ES_WE_p2p)
    [Gsess_r1, Sess_r1] = findgroups(T_ratio_ES_WE_p2p.Session);
    r1_mean = splitapply(@mean, T_ratio_ES_WE_p2p.Ratio, Gsess_r1);
    r1_std  = splitapply(@std,  T_ratio_ES_WE_p2p.Ratio, Gsess_r1);
    T_ES_WE_bySess_p2p = table(Sess_r1, r1_mean, r1_std, ...
                               'VariableNames',{'Session','MeanRatio','StdRatio'});
    T_ES_WE_bySess_p2p = innerjoin(T_ES_WE_bySess_p2p, speed_map, 'Keys','Session');
else
    T_ES_WE_bySess_p2p = table(cell(0,1), [], [], [], ...
                               'VariableNames',{'Session','MeanRatio','StdRatio','Speed'});
end

% SC/WE
if ~isempty(T_ratio_SC_WE_p2p)
    [Gsess_r2, Sess_r2] = findgroups(T_ratio_SC_WE_p2p.Session);
    r2_mean = splitapply(@mean, T_ratio_SC_WE_p2p.Ratio, Gsess_r2);
    r2_std  = splitapply(@std,  T_ratio_SC_WE_p2p.Ratio, Gsess_r2);
    T_SC_WE_bySess_p2p = table(Sess_r2, r2_mean, r2_std, ...
                               'VariableNames',{'Session','MeanRatio','StdRatio'});
    T_SC_WE_bySess_p2p = innerjoin(T_SC_WE_bySess_p2p, speed_map, 'Keys','Session');
else
    T_SC_WE_bySess_p2p = table(cell(0,1), [], [], [], ...
                               'VariableNames',{'Session','MeanRatio','StdRatio','Speed'});
end

% WE/SC
if ~isempty(T_ratio_WE_SC_p2p)
    [Gsess_r3, Sess_r3] = findgroups(T_ratio_WE_SC_p2p.Session);
    r3_mean = splitapply(@mean, T_ratio_WE_SC_p2p.Ratio, Gsess_r3);
    r3_std  = splitapply(@std,  T_ratio_WE_SC_p2p.Ratio, Gsess_r3);
    T_WE_SC_bySess_p2p = table(Sess_r3, r3_mean, r3_std, ...
                               'VariableNames',{'Session','MeanRatio','StdRatio'});
    T_WE_SC_bySess_p2p = innerjoin(T_WE_SC_bySess_p2p, speed_map, 'Keys','Session');
else
    T_WE_SC_bySess_p2p = table(cell(0,1), [], [], [], ...
                               'VariableNames',{'Session','MeanRatio','StdRatio','Speed'});
end

[sp_r1,ord_r1] = sort(T_ES_WE_bySess_p2p.Speed);
[sp_r2,ord_r2] = sort(T_SC_WE_bySess_p2p.Speed);
[sp_r3,ord_r3] = sort(T_WE_SC_bySess_p2p.Speed);

figure('Color','w','Position',[100 100 1200 800], ...
       'Name','P2P Speed-wise Ratio (ES/WE, SC/WE, WE/SC)','NumberTitle','off');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(1); hold(ax1,'on'); grid(ax1,'on'); box(ax1,'on');
if ~isempty(sp_r1)
    errorbar(ax1, sp_r1, T_ES_WE_bySess_p2p.MeanRatio(ord_r1), T_ES_WE_bySess_p2p.StdRatio(ord_r1), '-o','LineWidth',1.8);
end
xlabel(ax1,'Speed (m/s)'); ylabel(ax1,'ES/WE (P2P)');
title(ax1,'ES/WE ratio (P2P)');

ax2 = nexttile(2); hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on');
if ~isempty(sp_r2)
    errorbar(ax2, sp_r2, T_SC_WE_bySess_p2p.MeanRatio(ord_r2), T_SC_WE_bySess_p2p.StdRatio(ord_r2), '-o','LineWidth',1.8);
end
xlabel(ax2,'Speed (m/s)'); ylabel(ax2,'SC/WE (P2P)');
title(ax2,'SC/WE ratio (P2P)');

ax3 = nexttile(3); hold(ax3,'on'); grid(ax3,'on'); box(ax3,'on');
if ~isempty(sp_r3)
    errorbar(ax3, sp_r3, T_WE_SC_bySess_p2p.MeanRatio(ord_r3), T_WE_SC_bySess_p2p.StdRatio(ord_r3), '-o','LineWidth',1.8);
end
xlabel(ax3,'Speed (m/s)'); ylabel(ax3,'WE/SC (P2P)');
title(ax3,'WE/SC ratio (P2P)');

disp('=== P2P 기반 speed-wise 분석 완료 ===');


%% 3) Offset(평균) 기준: 속도(speed-wise) 분석 (WE / ES / SC + 비율)
clc;

%% 3-1) WE / ES / SC 각자 speed-wise Offset mean±SD

if ~isempty(T_WE_off)
    [Gsess_we_off, SessList_we_off] = findgroups(T_WE_off.Session);
    we_off_mean_s = splitapply(@mean, T_WE_off.WE_Mean, Gsess_we_off);
    we_off_std_s  = splitapply(@std,  T_WE_off.WE_Mean, Gsess_we_off);
    T_WE_bySess_off = table(SessList_we_off, we_off_mean_s, we_off_std_s, ...
                            'VariableNames',{'Session','MeanVal','StdVal'});
    T_WE_bySess_off = innerjoin(T_WE_bySess_off, speed_map, 'Keys','Session');
else
    T_WE_bySess_off = table(cell(0,1), [], [], [], ...
                            'VariableNames',{'Session','MeanVal','StdVal','Speed'});
end

if ~isempty(T_ES_off)
    [Gsess_es_off, SessList_es_off] = findgroups(T_ES_off.Session);
    es_off_mean_s = splitapply(@mean, T_ES_off.ES_Mean, Gsess_es_off);
    es_off_std_s  = splitapply(@std,  T_ES_off.ES_Mean, Gsess_es_off);
    T_ES_bySess_off = table(SessList_es_off, es_off_mean_s, es_off_std_s, ...
                            'VariableNames',{'Session','MeanVal','StdVal'});
    T_ES_bySess_off = innerjoin(T_ES_bySess_off, speed_map, 'Keys','Session');
else
    T_ES_bySess_off = table(cell(0,1), [], [], [], ...
                            'VariableNames',{'Session','MeanVal','StdVal','Speed'});
end

if ~isempty(T_SC_off)
    [Gsess_sc_off, SessList_sc_off] = findgroups(T_SC_off.Session);
    sc_off_mean_s = splitapply(@mean, T_SC_off.SC_Mean, Gsess_sc_off);
    sc_off_std_s  = splitapply(@std,  T_SC_off.SC_Mean, Gsess_sc_off);
    T_SC_bySess_off = table(SessList_sc_off, sc_off_mean_s, sc_off_std_s, ...
                            'VariableNames',{'Session','MeanVal','StdVal'});
    T_SC_bySess_off = innerjoin(T_SC_bySess_off, speed_map, 'Keys','Session');
else
    T_SC_bySess_off = table(cell(0,1), [], [], [], ...
                            'VariableNames',{'Session','MeanVal','StdVal','Speed'});
end

[sp_we_off,ord_we_off_s] = sort(T_WE_bySess_off.Speed);
[sp_es_off,ord_es_off_s] = sort(T_ES_bySess_off.Speed);
[sp_sc_off,ord_sc_off_s] = sort(T_SC_bySess_off.Speed);

figure('Color','w','Position',[100 100 1200 800], ...
       'Name','Offset Speed-wise mean±SD (WE / ES / SC)','NumberTitle','off');
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(1); hold(ax1,'on'); grid(ax1,'on'); box(ax1,'on');
if ~isempty(sp_we_off)
    errorbar(ax1, sp_we_off, T_WE_bySess_off.MeanVal(ord_we_off_s), T_WE_bySess_off.StdVal(ord_we_off_s), '-o','LineWidth',1.8);
end
xlabel(ax1,'Speed (m/s)'); ylabel(ax1,'WE Mean (deg)');
title(ax1,'WE: Speed-wise Offset');

ax2 = nexttile(2); hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on');
if ~isempty(sp_es_off)
    errorbar(ax2, sp_es_off, T_ES_bySess_off.MeanVal(ord_es_off_s), T_ES_bySess_off.StdVal(ord_es_off_s), '-o','LineWidth',1.8);
end
xlabel(ax2,'Speed (m/s)'); ylabel(ax2,'ES Mean (deg)');
title(ax2,'ES: Speed-wise Offset');

ax3 = nexttile(3); hold(ax3,'on'); grid(ax3,'on'); box(ax3,'on');
if ~isempty(sp_sc_off)
    errorbar(ax3, sp_sc_off, T_SC_bySess_off.MeanVal(ord_sc_off_s), T_SC_bySess_off.StdVal(ord_sc_off_s), '-o','LineWidth',1.8);
end
xlabel(ax3,'Speed (m/s)'); ylabel(ax3,'SC Mean (deg)');
title(ax3,'SC: Speed-wise Offset');

%% 3-2) Offset 비율: ES/WE, WE/SC (Speed-wise)

clc;

T_ratio_ES_WE_off = table();
T_ratio_WE_SC_off = table();

if ~isempty(T_ES_off) && ~isempty(T_WE_off)
    [Gtrial_es_off, Sub_es_off, Sess_es_off, Day_es_off] = findgroups(T_ES_off.Subject, T_ES_off.Session, T_ES_off.Day);
    es_mean_trial_off = splitapply(@mean, T_ES_off.ES_Mean, Gtrial_es_off);
    T_ES_trial_off = table(Sub_es_off, Sess_es_off, Day_es_off, es_mean_trial_off, ...
                           'VariableNames',{'Subject','Session','Day','ES_MeanVal'});

    [Gtrial_we_off, Sub_we_off2, Sess_we_off2, Day_we_off] = findgroups(T_WE_off.Subject, T_WE_off.Session, T_WE_off.Day);
    we_mean_trial_off = splitapply(@mean, T_WE_off.WE_Mean, Gtrial_we_off);
    T_WE_trial_off = table(Sub_we_off2, Sess_we_off2, Day_we_off, we_mean_trial_off, ...
                           'VariableNames',{'Subject','Session','Day','WE_MeanVal'});

    T_ratio_ES_WE_off = innerjoin(T_ES_trial_off, T_WE_trial_off, 'Keys',{'Subject','Session','Day'});
    T_ratio_ES_WE_off.Ratio = T_ratio_ES_WE_off.ES_MeanVal ./ T_ratio_ES_WE_off.WE_MeanVal;
    T_ratio_ES_WE_off = innerjoin(T_ratio_ES_WE_off, speed_map, 'Keys','Session');
end

if ~isempty(T_SC_off) && ~isempty(T_WE_off)
    [Gtrial_sc_off, Sub_sc_off, Sess_sc_off, Day_sc_off] = findgroups(T_SC_off.Subject, T_SC_off.Session, T_SC_off.Day);
    sc_mean_trial_off = splitapply(@mean, T_SC_off.SC_Mean, Gtrial_sc_off);
    T_SC_trial_off = table(Sub_sc_off, Sess_sc_off, Day_sc_off, sc_mean_trial_off, ...
                           'VariableNames',{'Subject','Session','Day','SC_MeanVal'});

    [Gtrial_we_off2, Sub_we_off3, Sess_we_off3, Day_we_off2] = findgroups(T_WE_off.Subject, T_WE_off.Session, T_WE_off.Day);
    we_mean_trial_off2 = splitapply(@mean, T_WE_off.WE_Mean, Gtrial_we_off2);
    T_WE_trial_off2 = table(Sub_we_off3, Sess_we_off3, Day_we_off2, we_mean_trial_off2, ...
                            'VariableNames',{'Subject','Session','Day','WE_MeanVal'});

    T_ratio_WE_SC_off = innerjoin(T_WE_trial_off2, T_SC_trial_off, 'Keys',{'Subject','Session','Day'});
    T_ratio_WE_SC_off.Ratio = T_ratio_WE_SC_off.WE_MeanVal ./ T_ratio_WE_SC_off.SC_MeanVal;
    T_ratio_WE_SC_off = innerjoin(T_ratio_WE_SC_off, speed_map, 'Keys','Session');
end

% ES/WE speed-wise
if ~isempty(T_ratio_ES_WE_off)
    [Gsess_r_off, Sess_r_off] = findgroups(T_ratio_ES_WE_off.Session);
    r_off_mean = splitapply(@mean, T_ratio_ES_WE_off.Ratio, Gsess_r_off);
    r_off_std  = splitapply(@std,  T_ratio_ES_WE_off.Ratio, Gsess_r_off);
    T_ES_WE_bySess_off = table(Sess_r_off, r_off_mean, r_off_std, ...
                               'VariableNames',{'Session','MeanRatio','StdRatio'});
    T_ES_WE_bySess_off = innerjoin(T_ES_WE_bySess_off, speed_map, 'Keys','Session');
else
    T_ES_WE_bySess_off = table(cell(0,1), [], [], [], ...
                               'VariableNames',{'Session','MeanRatio','StdRatio','Speed'});
end

% WE/SC speed-wise
if ~isempty(T_ratio_WE_SC_off)
    [Gsess_r2_off, Sess_r2_off] = findgroups(T_ratio_WE_SC_off.Session);
    r2_off_mean = splitapply(@mean, T_ratio_WE_SC_off.Ratio, Gsess_r2_off);
    r2_off_std  = splitapply(@std,  T_ratio_WE_SC_off.Ratio, Gsess_r2_off);
    T_WE_SC_bySess_off = table(Sess_r2_off, r2_off_mean, r2_off_std, ...
                               'VariableNames',{'Session','MeanRatio','StdRatio'});
    T_WE_SC_bySess_off = innerjoin(T_WE_SC_bySess_off, speed_map, 'Keys','Session');
else
    T_WE_SC_bySess_off = table(cell(0,1), [], [], [], ...
                               'VariableNames',{'Session','MeanRatio','StdRatio','Speed'});
end

[sp_r_off, ord_r_off]   = sort(T_ES_WE_bySess_off.Speed);
[sp_r2_off, ord_r2_off] = sort(T_WE_SC_bySess_off.Speed);

figure('Color','w','Position',[100 100 1200 600], ...
       'Name','Offset Speed-wise Ratio (ES/WE, WE/SC)','NumberTitle','off');
tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(1); hold(ax1,'on'); grid(ax1,'on'); box(ax1,'on');
if ~isempty(sp_r_off)
    errorbar(ax1, sp_r_off, T_ES_WE_bySess_off.MeanRatio(ord_r_off), T_ES_WE_bySess_off.StdRatio(ord_r_off), '-o','LineWidth',1.8);
end
xlabel(ax1,'Speed (m/s)'); ylabel(ax1,'ES/WE (Offset)');
title(ax1,'ES/WE ratio (Offset)');

ax2 = nexttile(2); hold(ax2,'on'); grid(ax2,'on'); box(ax2,'on');
if ~isempty(sp_r2_off)
    errorbar(ax2, sp_r2_off, T_WE_SC_bySess_off.MeanRatio(ord_r2_off), T_WE_SC_bySess_off.StdRatio(ord_r2_off), '-o','LineWidth',1.8);
end
xlabel(ax2,'Speed (m/s)'); ylabel(ax2,'WE/SC (Offset)');
title(ax2,'WE/SC ratio (Offset)');

disp('=== Offset 기반 speed-wise 분석 완료 ===');
