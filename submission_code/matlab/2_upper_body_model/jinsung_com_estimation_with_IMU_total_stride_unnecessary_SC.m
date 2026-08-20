% ================================================================
% A) DATA 준비: 모든 subj/session/day/stride 수집 → 시간정규화 → 집계
%     + 상대가속도(WE/ES/SC) "참값" 계산(관절 pos 2차미분) 및 "모델"과 비교용 행렬 추가
%     + Sacrum 분해: sacrum_true ≈ wrist_true + WE_true + ES_true + SC_true
%     + OFFSET_MODE에 'gravity' 추가 (stride별 vWE.offset_index/offset_rad 사용)
%     + ES/SC 오프셋 = es_offset_scaling_factor/sc_offset_scaling_factor * off_we
%     + 모든 patch는 legend에서 제외(자동 data1,2,… 제거)
% ================================================================
clc; % close all;

% ---------------- 사용자 옵션 ----------------
fs  = 100;             % Hz
dt  = 1/fs;
ES_SCALE = 0.497;
SC_SCALE = 0.12;       % SC = SC_SCALE * WE

% IMU 가속도 프레임: 'global' | 'local'  (pos_x/z는 global 기준이라 global 권장)
ACC_FRAME = 'global';

% 오프셋 옵션
% 'gravity' : vWE.offset_index / vWE.offset_rad를 사용해 stride별 off_we를 설정
OFFSET_MODE   = 'gravity';           % 'auto' | 'off' | 'manual' | 'gravity'
MANUAL_OFF_WE = deg2rad(0.0);     % manual일 때 사용 (rad)
MANUAL_OFF_ES = deg2rad(0.0);
MANUAL_OFF_SC = deg2rad(0.0);
% ES/SC 오프셋 스케일(we 오프셋의 배수)
es_offset_scaling_factor = 0.76;
sc_offset_scaling_factor = -1.2;

APPLY_OFFSET_TO_FORWARD = true;   % 전파식 각도에 오프셋 적용 여부

% 처리 대상 필터 (빈 배열이면 전부 처리)
target_subjects = {};             % 예: {'S002','S003'}  비우면 전체
target_sessions = {};        % 예: {'ss2','ss1'}  비우면 전체
target_days     = {};              % 예: {'Day1','Day3'}  비우면 전체
stride_range    = [];             % 예: [3 10]  비우면 Day 내 전체 stride

% 시간정규화 샘플
Nnorm  = 200;                     % 0~100% 구간을 Nnorm개로 리샘플
phase_tgt = linspace(0,1,Nnorm);
ph_pct    = linspace(0,100,Nnorm);

% ---------------- 사전 체크 ----------------
if ~(exist('stride_kinematics_arm','var')==1 && exist('stride_kinematics_arm_vector','var')==1)
    error('stride_kinematics_arm / stride_kinematics_arm_vector 가 필요합니다.');
end
if ~isfield(stride_kinematics_arm,'L_Wrist')
    error('stride_kinematics_arm.L_Wrist 가 없습니다.');
end

% ---------------- 집계 버퍼 (모델/참값 모두 저장) ----------------
% (1) 모델: wrist IMU, inc(WE/ES/SC), cumulative(shoulder), sacrum IMU
M_wrist_x = []; M_wrist_z = [];
M_inc_we_x = []; M_inc_we_z = [];
M_inc_es_x = []; M_inc_es_z = [];
M_inc_sc_x = []; M_inc_sc_z = [];
M_cum_x    = []; M_cum_z    = [];
M_sac_x    = []; M_sac_z    = [];

% (2) 참값: joint true acc 및 상대가속도
M_wrist_true_x = []; M_wrist_true_z = [];
M_elbow_true_x = []; M_elbow_true_z = [];
M_shldr_true_x = []; M_shldr_true_z = [];
M_inc_we_true_x = []; M_inc_we_true_z = [];
M_inc_es_true_x = []; M_inc_es_true_z = [];
M_inc_sc_true_x = []; M_inc_sc_true_z = [];

% (3) 합성(분해) 비교: sacrum_true vs wrist_true + inc_WE_true + inc_ES_true + inc_SC_true
M_sum_comp_x = []; M_sum_comp_z = [];

% 카운트
n_stride_total = 0; n_stride_used = 0; n_stride_skipped = 0;
n_stride_with_sacrum = 0;

% ---------------- 루프: Subject / Session / Day ----------------
all_subj = fieldnames(stride_kinematics_arm.L_Wrist).';
if ~isempty(target_subjects), all_subj = all_subj(ismember(all_subj, target_subjects)); end

for iS = 1:numel(all_subj)
    subj = all_subj{iS};
    if ~isfield(stride_kinematics_arm.L_Wrist, subj), continue; end

    all_sess = fieldnames(stride_kinematics_arm.L_Wrist.(subj)).';
    if ~isempty(target_sessions), all_sess = all_sess(ismember(all_sess, target_sessions)); end

    for iSe = 1:numel(all_sess)
        sess = all_sess{iSe};
        if ~isfield(stride_kinematics_arm.L_Wrist.(subj), sess), continue; end

        all_days = fieldnames(stride_kinematics_arm.L_Wrist.(subj).(sess)).';
        isDay = ~cellfun('isempty', regexp(all_days,'^Day','once'));
        all_days = all_days(isDay);
        if ~isempty(target_days), all_days = all_days(ismember(all_days, target_days)); end

        for iD = 1:numel(all_days)
            day = all_days{iD};

            % leaf
            if ~isfield(stride_kinematics_arm.L_Wrist.(subj).(sess), day), continue; end
            if ~isfield(stride_kinematics_arm.L_Elbow.(subj).(sess), day), continue; end
            if ~isfield(stride_kinematics_arm.L_Shoulder.(subj).(sess), day), continue; end
            wLeaf = stride_kinematics_arm.L_Wrist.(subj).(sess).(day);
            eLeaf = stride_kinematics_arm.L_Elbow.(subj).(sess).(day);
            sLeaf = stride_kinematics_arm.L_Shoulder.(subj).(sess).(day);

            cLeaf = [];
            if isfield(stride_kinematics_arm,'sacrum') && ...
               isfield(stride_kinematics_arm.sacrum, subj) && ...
               isfield(stride_kinematics_arm.sacrum.(subj), sess) && ...
               isfield(stride_kinematics_arm.sacrum.(subj).(sess), day)
                cLeaf = stride_kinematics_arm.sacrum.(subj).(sess).(day);
            end

            % vector lengths (WE/ES/SC) + gravity offsets 원천(vWE)
            haveVec = true;
            if isfield(stride_kinematics_arm_vector,'L_Wrist_to_L_Elbow') && ...
               isfield(stride_kinematics_arm_vector.L_Wrist_to_L_Elbow, subj) && ...
               isfield(stride_kinematics_arm_vector.L_Wrist_to_L_Elbow.(subj), sess) && ...
               isfield(stride_kinematics_arm_vector.L_Wrist_to_L_Elbow.(subj).(sess), day)
                vWE = stride_kinematics_arm_vector.L_Wrist_to_L_Elbow.(subj).(sess).(day);
            else
                haveVec = false;
            end
            if isfield(stride_kinematics_arm_vector,'L_Elbow_to_L_Shoulder') && ...
               isfield(stride_kinematics_arm_vector.L_Elbow_to_L_Shoulder, subj) && ...
               isfield(stride_kinematics_arm_vector.L_Elbow_to_L_Shoulder.(subj), sess) && ...
               isfield(stride_kinematics_arm_vector.L_Elbow_to_L_Shoulder.(subj).(sess), day)
                vES = stride_kinematics_arm_vector.L_Elbow_to_L_Shoulder.(subj).(sess).(day);
            else
                haveVec = false;
            end
            if isfield(stride_kinematics_arm_vector,'L_Shoulder_to_sacrum') && ...
               isfield(stride_kinematics_arm_vector.L_Shoulder_to_sacrum, subj) && ...
               isfield(stride_kinematics_arm_vector.L_Shoulder_to_sacrum.(subj), sess) && ...
               isfield(stride_kinematics_arm_vector.L_Shoulder_to_sacrum.(subj).(sess), day)
                vSC = stride_kinematics_arm_vector.L_Shoulder_to_sacrum.(subj).(sess).(day);
            else
                haveVec = false;
            end
            if ~haveVec, continue; end

            % stride 개수
            if ~isfield(wLeaf,'imu_gyro_local_y') || ~iscell(wLeaf.imu_gyro_local_y) || isempty(wLeaf.imu_gyro_local_y)
                continue;
            end
            nStride = numel(wLeaf.imu_gyro_local_y);

            % stride 범위
            if isempty(stride_range)
                s1=1; s2=nStride;
            else
                s1 = max(1, stride_range(1)); s2 = min(nStride, stride_range(2));
                if s1> s2, continue; end
            end

            % ================= STRIDE LOOP =================
            for s = s1:s2
                n_stride_total = n_stride_total + 1;

                % 필수 위치/gyro/acc 체크
                okW = isfield(wLeaf,'pos_x') && iscell(wLeaf.pos_x) && numel(wLeaf.pos_x)>=s && ~isempty(wLeaf.pos_x{s}) && ...
                      isfield(wLeaf,'pos_z') && iscell(wLeaf.pos_z) && numel(wLeaf.pos_z)>=s && ~isempty(wLeaf.pos_z{s});
                okE = isfield(eLeaf,'pos_x') && iscell(eLeaf.pos_x) && numel(eLeaf.pos_x)>=s && ~isempty(eLeaf.pos_x{s}) && ...
                      isfield(eLeaf,'pos_z') && iscell(eLeaf.pos_z) && numel(eLeaf.pos_z)>=s && ~isempty(eLeaf.pos_z{s});
                okS = isfield(sLeaf,'pos_x') && iscell(sLeaf.pos_x) && numel(sLeaf.pos_x)>=s && ~isempty(sLeaf.pos_x{s}) && ...
                      isfield(sLeaf,'pos_z') && iscell(sLeaf.pos_z) && numel(sLeaf.pos_z)>=s && ~isempty(sLeaf.pos_z{s});
                okG = isfield(wLeaf,'imu_gyro_local_y') && iscell(wLeaf.imu_gyro_local_y) && numel(wLeaf.imu_gyro_local_y)>=s && ~isempty(wLeaf.imu_gyro_local_y{s});
                if ~(okW && okE && okS && okG)
                    n_stride_skipped = n_stride_skipped + 1; continue;
                end

                wX = wLeaf.pos_x{s}(:);  wZ = wLeaf.pos_z{s}(:);
                eX = eLeaf.pos_x{s}(:);  eZ = eLeaf.pos_z{s}(:);
                sX = sLeaf.pos_x{s}(:);  sZ = sLeaf.pos_z{s}(:);
                gy = wLeaf.imu_gyro_local_y{s}(:);

                % Wrist IMU acc (프레임)
                if strcmpi(ACC_FRAME,'global')
                    haveAX = isfield(wLeaf,'imu_acc_global_x') && iscell(wLeaf.imu_acc_global_x) && numel(wLeaf.imu_acc_global_x)>=s && ~isempty(wLeaf.imu_acc_global_x{s});
                    haveAZ = isfield(wLeaf,'imu_acc_global_z') && iscell(wLeaf.imu_acc_global_z) && numel(wLeaf.imu_acc_global_z)>=s && ~isempty(wLeaf.imu_acc_global_z{s});
                    if ~(haveAX && haveAZ), n_stride_skipped = n_stride_skipped + 1; continue; end
                    ax_w = wLeaf.imu_acc_global_x{s}(:);
                    az_w = wLeaf.imu_acc_global_z{s}(:);
                else
                    haveAX = isfield(wLeaf,'imu_acc_local_x') && iscell(wLeaf.imu_acc_local_x) && numel(wLeaf.imu_acc_local_x)>=s && ~isempty(wLeaf.imu_acc_local_x{s});
                    haveAZ = isfield(wLeaf,'imu_acc_local_z') && iscell(wLeaf.imu_acc_local_z) && numel(wLeaf.imu_acc_local_z)>=s && ~isempty(wLeaf.imu_acc_local_z{s});
                    if ~(haveAX && haveAZ), n_stride_skipped = n_stride_skipped + 1; continue; end
                    ax_w = wLeaf.imu_acc_local_x{s}(:);
                    az_w = wLeaf.imu_acc_local_z{s}(:);
                end

                % sacrum IMU (optional)
                ax_sac = []; az_sac = [];
                if ~isempty(cLeaf)
                    if strcmpi(ACC_FRAME,'global')
                        if isfield(cLeaf,'imu_acc_global_x') && iscell(cLeaf.imu_acc_global_x) && numel(cLeaf.imu_acc_global_x)>=s && ~isempty(cLeaf.imu_acc_global_x{s})
                            ax_sac = cLeaf.imu_acc_global_x{s}(:);
                        end
                        if isfield(cLeaf,'imu_acc_global_z') && iscell(cLeaf.imu_acc_global_z) && numel(cLeaf.imu_acc_global_z)>=s && ~isempty(cLeaf.imu_acc_global_z{s})
                            az_sac = cLeaf.imu_acc_global_z{s}(:);
                        end
                    else
                        if isfield(cLeaf,'imu_acc_local_x') && iscell(cLeaf.imu_acc_local_x) && numel(cLeaf.imu_acc_local_x)>=s && ~isempty(cLeaf.imu_acc_local_x{s})
                            ax_sac = cLeaf.imu_acc_local_x{s}(:);
                        end
                        if isfield(cLeaf,'imu_acc_local_z') && iscell(cLeaf.imu_acc_local_z) && numel(cLeaf.imu_acc_local_z)>=s && ~isempty(cLeaf.imu_acc_local_z{s})
                            az_sac = cLeaf.imu_acc_local_z{s}(:);
                        end
                    end
                end

                % 길이(해당 stride 평균)
                okLwe = isfield(vWE,'length_xz') && isfield(vWE.length_xz,'pos') && iscell(vWE.length_xz.pos) && numel(vWE.length_xz.pos)>=s && ~isempty(vWE.length_xz.pos{s});
                okLes = isfield(vES,'length_xz') && isfield(vES.length_xz,'pos') && iscell(vES.length_xz.pos) && numel(vES.length_xz.pos)>=s && ~isempty(vES.length_xz.pos{s});
                okLsc = isfield(vSC,'length_xz') && isfield(vSC.length_xz,'pos') && iscell(vSC.length_xz.pos) && numel(vSC.length_xz.pos)>=s && ~isempty(vSC.length_xz.pos{s});
                if ~(okLwe && okLes && okLsc), n_stride_skipped = n_stride_skipped + 1; continue; end
                L_we = mean(vWE.length_xz.pos{s}(:),'omitnan');
                L_es = mean(vES.length_xz.pos{s}(:),'omitnan');
                L_sc = mean(vSC.length_xz.pos{s}(:),'omitnan');

                % 공통 길이 N
                Ls = [numel(wX), numel(wZ), numel(eX), numel(eZ), numel(sX), numel(sZ), numel(gy), numel(ax_w), numel(az_w)];
                if ~isempty(ax_sac), Ls(end+1)=numel(ax_sac); end
                if ~isempty(az_sac), Ls(end+1)=numel(az_sac); end
                N = min(Ls);
                if N < 5, n_stride_skipped = n_stride_skipped + 1; continue; end

                % crop & 시간축
                wX=wX(1:N); wZ=wZ(1:N); eX=eX(1:N); eZ=eZ(1:N); sX=sX(1:N); sZ=sZ(1:N);
                gy=gy(1:N);  ax_w=ax_w(1:N); az_w=az_w(1:N);
                if ~isempty(ax_sac), ax_sac = ax_sac(1:N); end
                if ~isempty(az_sac), az_sac = az_sac(1:N); end
                t  = (0:N-1)'/fs;

                % gyro 바이어스 제거 + 결측 보간
                idxF = isfinite(gy);
                if any(idxF), gy = gy - median(gy(idxF)); end
                if any(~idxF)
                    if nnz(idxF)>=2
                        gy(~idxF) = interp1(t(idxF), gy(idxF), t(~idxF), 'linear', 'extrap');
                    else
                        gy(~idxF) = 0;
                    end
                end

                % Pred 각도: WE(gyro 적분) / ES(scale) / SC(scale)
                th_we_pred = cumtrapz(t, gy); th_we_pred = th_we_pred - th_we_pred(1);
                th_es_pred = ES_SCALE * th_we_pred;
                th_sc_pred = SC_SCALE * th_we_pred;

                % True 각 (시상면) — sacrum pos 없음 → SC true 각은 없음
                th_we_true = atan2(eZ - wZ, eX - wX);
                th_es_true = atan2(sZ - eZ, sX - eX);

                % ---------------- 오프셋 결정 ----------------
                switch lower(OFFSET_MODE)
                    case 'off'
                        off_we = 0.0;
                        off_es = 0.0;
                        off_sc = 0.0;

                    case 'manual'
                        off_we = MANUAL_OFF_WE;
                        off_es = MANUAL_OFF_ES;
                        off_sc = MANUAL_OFF_SC;

                    case 'auto'
                        d_we = th_we_true - th_we_pred;
                        d_es = th_es_true - th_es_pred;
                        off_we = atan2(nanmean(sin(d_we)), nanmean(cos(d_we)));
                        % ES/SC는 스케일링 팩터 기준(기본 1.0 → off_es≈off_we)
                        off_es = es_offset_scaling_factor * off_we;
                        off_sc = sc_offset_scaling_factor * off_we;

                    case 'gravity'
                        % vWE.offset_index / vWE.offset_rad에서 stride s에 해당하는 값을 읽기
                        k0 = NaN; target_rad = NaN;
                        if isfield(vWE,'offset_index')
                            if iscell(vWE.offset_index) && numel(vWE.offset_index)>=s && ~isempty(vWE.offset_index{s})
                                k0 = vWE.offset_index{s}(1);
                            elseif isnumeric(vWE.offset_index) && numel(vWE.offset_index)>=s
                                k0 = vWE.offset_index(s);
                            end
                        end
                        if isfield(vWE,'offset_rad')
                            if iscell(vWE.offset_rad) && numel(vWE.offset_rad)>=s && ~isempty(vWE.offset_rad{s})
                                target_rad = vWE.offset_rad{s}(1);
                            elseif isnumeric(vWE.offset_rad) && numel(vWE.offset_rad)>=s
                                target_rad = vWE.offset_rad(s);
                            end
                        end
                        if ~isnan(k0) && ~isnan(target_rad)
                            k0 = max(1, min(N, round(k0)));
                            % "그 index에서의 각을 target_rad로 맞추도록" 오프셋 결정
                            off_we = atan2(sin(target_rad - th_we_pred(k0)), cos(target_rad - th_we_pred(k0)));
                        else
                            % fallback: auto
                            d_we = th_we_true - th_we_pred;
                            off_we = atan2(nanmean(sin(d_we)), nanmean(cos(d_we)));
                        end
                        % ES/SC는 we 오프셋의 배수
                        off_es = es_offset_scaling_factor * off_we;
                        off_sc = sc_offset_scaling_factor * off_we;

                    otherwise
                        warning('알 수 없는 OFFSET_MODE=%s → auto 사용', OFFSET_MODE);
                        d_we = th_we_true - th_we_pred;
                        off_we = atan2(nanmean(sin(d_we)), nanmean(cos(d_we)));
                        off_es = es_offset_scaling_factor * off_we;
                        off_sc = sc_offset_scaling_factor * off_we;
                end

                % 전파식 각도 (오프셋 적용 여부)
                if APPLY_OFFSET_TO_FORWARD
                    th_we_fd = th_we_pred + off_we;
                    th_es_fd = th_es_pred + off_es;
                    th_sc_fd = th_sc_pred + off_sc;
                else
                    th_we_fd = th_we_pred;
                    th_es_fd = th_es_pred;
                    th_sc_fd = th_sc_pred;
                end

                % 각속/각가속도
                om_we = gy;                 al_we = gradient(om_we, dt);
                om_es = ES_SCALE * om_we;   al_es = ES_SCALE * al_we;
                om_sc = SC_SCALE * om_we;   al_sc = SC_SCALE * al_we;

                % (모델) 상대가속도 항 (2D 근사, X/Z 동일각 사용)
                rddx_we = -L_we .* ( cos(th_we_fd).*(om_we.^2) + sin(th_we_fd).*al_we );
                rddz_we = -L_we .* ( sin(th_we_fd).*(om_we.^2) - cos(th_we_fd).*al_we );

                rddx_es = -L_es .* ( cos(th_es_fd).*(om_es.^2) + sin(th_es_fd).*al_es );
                rddz_es = -L_es .* ( sin(th_es_fd).*(om_es.^2) - cos(th_es_fd).*al_es );

                % SC inc (model)
                rddx_sc = -L_sc .* ( cos(th_sc_fd).*(om_sc.^2) + sin(th_sc_fd).*al_sc );
                rddz_sc = -L_sc .* ( sin(th_sc_fd).*(om_sc.^2) - cos(th_sc_fd).*al_sc );

                % (모델) 누적 전파(어깨까지만)
                ax_elbow_m    = ax_w + rddx_we;
                az_elbow_m    = az_w + rddz_we;
                ax_shoulder_m = ax_elbow_m + rddx_es;
                az_shoulder_m = az_elbow_m + rddz_es;

                % ------------------ (참값) joint 가속도: pos 2차 미분 ------------------
                vx_w = gradient(wX, dt); ax_w_t = gradient(vx_w, dt);
                vz_w = gradient(wZ, dt); az_w_t = gradient(vz_w, dt);

                vx_e = gradient(eX, dt); ax_e_t = gradient(vx_e, dt);
                vz_e = gradient(eZ, dt); az_e_t = gradient(vz_e, dt);

                vx_sj = gradient(sX, dt); ax_s_t = gradient(vx_sj, dt);
                vz_sj = gradient(sZ, dt); az_s_t = gradient(vz_sj, dt);

                % (참값) 상대가속도
                inc_we_tx = ax_e_t - ax_w_t;   inc_we_tz = az_e_t - az_w_t;   % E - W
                inc_es_tx = ax_s_t - ax_e_t;   inc_es_tz = az_s_t - az_e_t;   % S - E

                % sacrum true(X/Z) = IMU (frame 일치 가정) → shoulder->sacrum "참값" 상대가속도
                if ~isempty(ax_sac) && ~isempty(az_sac) && numel(ax_sac)==N && numel(az_sac)==N
                    inc_sc_tx = ax_sac - ax_s_t;   % C - S
                    inc_sc_tz = az_sac - az_s_t;
                    have_sc = true;
                else
                    inc_sc_tx = nan(N,1); inc_sc_tz = nan(N,1);
                    have_sc = false;
                end

                % ------------------ 시간정규화(0~1 → Nnorm) ------------------
                phase_src = linspace(0,1,N);

                % 모델 (wrist IMU, inc, cumulative, sacrum IMU)
                wrx_m = interp1(phase_src, ax_w,         phase_tgt, 'linear','extrap');
                wrz_m = interp1(phase_src, az_w,         phase_tgt, 'linear','extrap');
                iwx_m = interp1(phase_src, rddx_we,      phase_tgt, 'linear','extrap');
                iwz_m = interp1(phase_src, rddz_we,      phase_tgt, 'linear','extrap');
                esx_m = interp1(phase_src, rddx_es,      phase_tgt, 'linear','extrap');
                esz_m = interp1(phase_src, rddz_es,      phase_tgt, 'linear','extrap');
                scx_m = interp1(phase_src, rddx_sc,      phase_tgt, 'linear','extrap');
                scz_m = interp1(phase_src, rddz_sc,      phase_tgt, 'linear','extrap');

                cmx_m = interp1(phase_src, ax_shoulder_m,phase_tgt, 'linear','extrap');
                cmz_m = interp1(phase_src, az_shoulder_m,phase_tgt, 'linear','extrap');

                if have_sc
                    scx_t = interp1(phase_src, ax_sac, phase_tgt, 'linear','extrap');
                    scz_t = interp1(phase_src, az_sac, phase_tgt, 'linear','extrap');
                else
                    scx_t = nan(1,Nnorm); scz_t = nan(1,Nnorm);
                end

                % 참값 (joint true acc)
                wtx = interp1(phase_src, ax_w_t, phase_tgt, 'linear','extrap');
                wtz = interp1(phase_src, az_w_t, phase_tgt, 'linear','extrap');
                etx = interp1(phase_src, ax_e_t, phase_tgt, 'linear','extrap');
                etz = interp1(phase_src, az_e_t, phase_tgt, 'linear','extrap');
                stx = interp1(phase_src, ax_s_t, phase_tgt, 'linear','extrap');
                stz = interp1(phase_src, az_s_t, phase_tgt, 'linear','extrap');

                % 참값 상대가속도
                iwx_t = interp1(phase_src, inc_we_tx, phase_tgt, 'linear','extrap');
                iwz_t = interp1(phase_src, inc_we_tz, phase_tgt, 'linear','extrap');
                esx_t = interp1(phase_src, inc_es_tx, phase_tgt, 'linear','extrap');
                esz_t = interp1(phase_src, inc_es_tz, phase_tgt, 'linear','extrap');
                scx_it = interp1(phase_src, inc_sc_tx, phase_tgt, 'linear','extrap');
                scz_it = interp1(phase_src, inc_sc_tz, phase_tgt, 'linear','extrap');

                % 합성: wrist_true + inc_WE_true + inc_ES_true + inc_SC_true
                sum_comp_x = wtx + iwx_t + esx_t + scx_it;
                sum_comp_z = wtz + iwz_t + esz_t + scz_it;

                % ------------------ 집계 행렬에 추가 ------------------
                % 모델
                M_wrist_x(end+1,:)  = wrx_m(:).';   M_wrist_z(end+1,:)  = wrz_m(:).';
                M_inc_we_x(end+1,:) = iwx_m(:).';   M_inc_we_z(end+1,:) = iwz_m(:).';
                M_inc_es_x(end+1,:) = esx_m(:).';   M_inc_es_z(end+1,:) = esz_m(:).';
                M_inc_sc_x(end+1,:) = scx_m(:).';   M_inc_sc_z(end+1,:) = scz_m(:).';
                M_cum_x(end+1,:)    = cmx_m(:).';   M_cum_z(end+1,:)    = cmz_m(:).';
                M_sac_x(end+1,:)    = scx_t(:).';   M_sac_z(end+1,:)    = scz_t(:).';

                % 참값
                M_wrist_true_x(end+1,:) = wtx(:).';  M_wrist_true_z(end+1,:) = wtz(:).';
                M_elbow_true_x(end+1,:) = etx(:).';  M_elbow_true_z(end+1,:) = etz(:).';
                M_shldr_true_x(end+1,:) = stx(:).';  M_shldr_true_z(end+1,:) = stz(:).';

                M_inc_we_true_x(end+1,:) = iwx_t(:).';  M_inc_we_true_z(end+1,:) = iwz_t(:).';
                M_inc_es_true_x(end+1,:) = esx_t(:).';  M_inc_es_true_z(end+1,:) = esz_t(:).';
                M_inc_sc_true_x(end+1,:) = scx_it(:).'; M_inc_sc_true_z(end+1,:) = scz_it(:).';

                % 합성
                M_sum_comp_x(end+1,:) = sum_comp_x(:).';
                M_sum_comp_z(end+1,:) = sum_comp_z(:).';

                n_stride_used = n_stride_used + 1;
                if any(isfinite(scx_t)) && any(isfinite(scz_t)), n_stride_with_sacrum = n_stride_with_sacrum + 1; end
            end % stride
        end % day
    end % sess
end % subj

fprintf('[INFO] total strides: %d, used: %d, skipped: %d, with-sacrum: %d\n', ...
    n_stride_total, n_stride_used, n_stride_skipped, n_stride_with_sacrum);

% ---------------- 평균/표준편차 ----------------
% 모델
mu_wx = nanmean(M_wrist_x,1); sd_wx = nanstd(M_wrist_x,0,1);
mu_wz = nanmean(M_wrist_z,1); sd_wz = nanstd(M_wrist_z,0,1);

mu_iwx = nanmean(M_inc_we_x,1); sd_iwx = nanstd(M_inc_we_x,0,1);
mu_iwz = nanmean(M_inc_we_z,1); sd_iwz = nanstd(M_inc_we_z,0,1);

mu_esx = nanmean(M_inc_es_x,1); sd_esx = nanstd(M_inc_es_x,0,1);
mu_esz = nanmean(M_inc_es_z,1); sd_esz = nanstd(M_inc_es_z,0,1);

mu_scincx = nanmean(M_inc_sc_x,1); sd_scincx = nanstd(M_inc_sc_x,0,1);
mu_scincz = nanmean(M_inc_sc_z,1); sd_scincz = nanstd(M_inc_sc_z,0,1);

mu_cmx = nanmean(M_cum_x,1);    sd_cmx = nanstd(M_cum_x,0,1);
mu_cmz = nanmean(M_cum_z,1);    sd_cmz = nanstd(M_cum_z,0,1);

mu_scx = nanmean(M_sac_x,1);    sd_scx = nanstd(M_sac_x,0,1);
mu_scz = nanmean(M_sac_z,1);    sd_scz = nanstd(M_sac_z,0,1);

% 참값
mu_wtx = nanmean(M_wrist_true_x,1); sd_wtx = nanstd(M_wrist_true_x,0,1);
mu_wtz = nanmean(M_wrist_true_z,1); sd_wtz = nanstd(M_wrist_true_z,0,1);

mu_iwx_t = nanmean(M_inc_we_true_x,1); sd_iwx_t = nanstd(M_inc_we_true_x,0,1);
mu_iwz_t = nanmean(M_inc_we_true_z,1); sd_iwz_t = nanstd(M_inc_we_true_z,0,1);

mu_esx_t = nanmean(M_inc_es_true_x,1); sd_esx_t = nanstd(M_inc_es_true_x,0,1);
mu_esz_t = nanmean(M_inc_es_true_z,1); sd_esz_t = nanstd(M_inc_es_true_z,0,1);

mu_scx_it = nanmean(M_inc_sc_true_x,1); sd_scx_it = nanstd(M_inc_sc_true_x,0,1);
mu_scz_it = nanmean(M_inc_sc_true_z,1); sd_scz_it = nanstd(M_inc_sc_true_z,0,1);

mu_sumx = nanmean(M_sum_comp_x,1); sd_sumx = nanstd(M_sum_comp_x,0,1);
mu_sumz = nanmean(M_sum_comp_z,1); sd_sumz = nanstd(M_sum_comp_z,0,1);

% ---------------- 정확도 수치 (RMSE / Corr) ----------------
vec = @(M) M(:);

% WE inc (model vs true)
rmse_we_x = sqrt(nanmean((vec(M_inc_we_x) - vec(M_inc_we_true_x)).^2));
rmse_we_z = sqrt(nanmean((vec(M_inc_we_z) - vec(M_inc_we_true_z)).^2));
cx = corrcoef([vec(M_inc_we_x), vec(M_inc_we_true_x)], 'Rows','pairwise'); c_we_x = cx(1,2);
cz = corrcoef([vec(M_inc_we_z), vec(M_inc_we_true_z)], 'Rows','pairwise'); c_we_z = cz(1,2);

% ES inc (model vs true)
rmse_es_x = sqrt(nanmean((vec(M_inc_es_x) - vec(M_inc_es_true_x)).^2));
rmse_es_z = sqrt(nanmean((vec(M_inc_es_z) - vec(M_inc_es_true_z)).^2));
cx = corrcoef([vec(M_inc_es_x), vec(M_inc_es_true_x)], 'Rows','pairwise'); c_es_x = cx(1,2);
cz = corrcoef([vec(M_inc_es_z), vec(M_inc_es_true_z)], 'Rows','pairwise'); c_es_z = cz(1,2);

% SC inc (model vs true)
rmse_sc_x = sqrt(nanmean((vec(M_inc_sc_x) - vec(M_inc_sc_true_x)).^2));
rmse_sc_z = sqrt(nanmean((vec(M_inc_sc_z) - vec(M_inc_sc_true_z)).^2));
cx = corrcoef([vec(M_inc_sc_x), vec(M_inc_sc_true_x)], 'Rows','pairwise'); c_sc_x = cx(1,2);
cz = corrcoef([vec(M_inc_sc_z), vec(M_inc_sc_true_z)], 'Rows','pairwise'); c_sc_z = cz(1,2);

% Sacrum 분해 일치도
rmse_comp_x = sqrt(nanmean((vec(M_sac_x) - vec(M_sum_comp_x)).^2));
rmse_comp_z = sqrt(nanmean((vec(M_sac_z) - vec(M_sum_comp_z)).^2));
cx = corrcoef([vec(M_sac_x), vec(M_sum_comp_x)], 'Rows','pairwise'); c_comp_x = cx(1,2);
cz = corrcoef([vec(M_sac_z), vec(M_sum_comp_z)], 'Rows','pairwise'); c_comp_z = cz(1,2);

fprintf('[ACC] WE inc — RMSE X=%.4f, Z=%.4f | Corr X=%.3f, Z=%.3f\n', rmse_we_x, rmse_we_z, c_we_x, c_we_z);
fprintf('[ACC] ES inc — RMSE X=%.4f, Z=%.4f | Corr X=%.3f, Z=%.3f\n', rmse_es_x, rmse_es_z, c_es_x, c_es_z);
fprintf('[ACC] SC inc — RMSE X=%.4f, Z=%.4f | Corr X=%.3f, Z=%.3f\n', rmse_sc_x, rmse_sc_z, c_sc_x, c_sc_z);
fprintf('[ACC] Sacrum comp — RMSE X=%.4f, Z=%.4f | Corr X=%.3f, Z=%.3f  (NwithSacrum=%d)\n', ...
    rmse_comp_x, rmse_comp_z, c_comp_x, c_comp_z, n_stride_with_sacrum);

% ================================================================
% C) 추가 요청 (수정): *모델* 누적 합 vs *Sacrum 참값* 비교 (nRMSE 포함)
%    - Error = (Model Sum) - (Sacrum True)
%    - (W_model + WE_model + ES_model)    = M_cum
%    - (W_model + WE_model + ES_model + SC_model) = M_cum + M_inc_sc
%    - Sacrum True  = M_sac
% ================================================================

% (1) Error: (W_model + WE_model + ES_model) vs Sacrum_true
% (W_model + WE_model + ES_model)는 모델 어깨 가속도(M_cum)와 같습니다.
err_2seg_model_x_vec = vec(M_cum_x - M_sac_x);
err_2seg_model_z_vec = vec(M_cum_z - M_sac_z);

mean_err_2seg_model_x = nanmean(err_2seg_model_x_vec);
std_err_2seg_model_x  = nanstd(err_2seg_model_x_vec, 0);
mean_err_2seg_model_z = nanmean(err_2seg_model_z_vec);
std_err_2seg_model_z  = nanstd(err_2seg_model_z_vec, 0);

% (W_model + WE_model + ES_model)의 RMSE (vs Sacrum_true)
rmse_2seg_model_x = sqrt(nanmean(err_2seg_model_x_vec.^2));
rmse_2seg_model_z = sqrt(nanmean(err_2seg_model_z_vec.^2));

% (2) Error: (W_model + WE_model + ES_model + SC_model) vs Sacrum_true
% 모델 누적합 + 모델 SCinc
M_sum_model_x = M_cum_x + M_inc_sc_x;
M_sum_model_z = M_cum_z + M_inc_sc_z;

err_3seg_model_x_vec = vec(M_sum_model_x - M_sac_x);
err_3seg_model_z_vec = vec(M_sum_model_z - M_sac_z);

mean_err_3seg_model_x = nanmean(err_3seg_model_x_vec);
std_err_3seg_model_x  = nanstd(err_3seg_model_x_vec, 0);
mean_err_3seg_model_z = nanmean(err_3seg_model_z_vec);
std_err_3seg_model_z  = nanstd(err_3seg_model_z_vec, 0);

% (W_model + WE_model + ES_model + SC_model)의 RMSE (vs Sacrum_true)
rmse_3seg_model_x = sqrt(nanmean(err_3seg_model_x_vec.^2));
rmse_3seg_model_z = sqrt(nanmean(err_3seg_model_z_vec.^2));

% ------------------ nRMSE (Normalized by Range) ------------------
true_sac_x_vec = vec(M_sac_x);
true_sac_z_vec = vec(M_sac_z);

% Sacrum 참값의 Max-Min Range 계산
range_sac_x = max(true_sac_x_vec, [], 'omitnan') - min(true_sac_x_vec, [], 'omitnan');
range_sac_z = max(true_sac_z_vec, [], 'omitnan') - min(true_sac_z_vec, [], 'omitnan');

% 0으로 나누는 오류 방지
if range_sac_x == 0, range_sac_x = 1; end
if range_sac_z == 0, range_sac_z = 1; end

% nRMSE 계산 (% 단위)
nrmse_2seg_model_x = (rmse_2seg_model_x / range_sac_x) * 100;
nrmse_2seg_model_z = (rmse_2seg_model_z / range_sac_z) * 100;
nrmse_3seg_model_x = (rmse_3seg_model_x / range_sac_x) * 100;
nrmse_3seg_model_z = (rmse_3seg_model_z / range_sac_z) * 100;

% ------------------ 결과 출력 ------------------
fprintf('\n--- 추가 분석 (True Sacrum IMU 대비 *모델* 누적합 Error) ---\n');
fprintf('[INFO] nRMSE normalized by Range(True Sacrum): X=%.2f (m/s^2), Z=%.2f (m/s^2)\n', range_sac_x, range_sac_z);
fprintf('[ERR] (W_model + WE_model + ES_model) vs Sacrum   — Mean±SD (X): %.4f ± %.4f (RMSE: %.4f, nRMSE: %.2f%%)\n', ...
    mean_err_2seg_model_x, std_err_2seg_model_x, rmse_2seg_model_x, nrmse_2seg_model_x);
fprintf('[ERR] (W_model + WE_model + ES_model) vs Sacrum   — Mean±SD (Z): %.4f ± %.4f (RMSE: %.4f, nRMSE: %.2f%%)\n', ...
    mean_err_2seg_model_z, std_err_2seg_model_z, rmse_2seg_model_z, nrmse_2seg_model_z);
fprintf('[ERR] (W_model + WE_model + ES_model + SC_model) vs Sacrum — Mean±SD (X): %.4f ± %.4f (RMSE: %.4f, nRMSE: %.2f%%)\n', ...
    mean_err_3seg_model_x, std_err_3seg_model_x, rmse_3seg_model_x, nrmse_3seg_model_x);
fprintf('[ERR] (W_model + WE_model + ES_model + SC_model) vs Sacrum — Mean±SD (Z): %.4f ± %.4f (RMSE: %.4f, nRMSE: %.2f%%)\n', ...
    mean_err_3seg_model_z, std_err_3seg_model_z, rmse_3seg_model_z, nrmse_3seg_model_z);
fprintf('-----------------------------------------------------\n\n');

%% 색상 정의
col_wrist    = [192 0   0  ]/255;   % wrist
col_we       = [228 172 41 ]/255;   % WE segment (model, increment)
col_es       = [21  107 177]/255;   % ES segment (model, increment)
col_sc       = [64  162 55 ]/255;   % SC segment (model, increment)
col_sac_est  = [112 48  160]/255;   % sacrum estimate (W+WE+ES+SC)
col_sac_true = [0   0   0  ];       % sacrum TRUE (검정)
col_sac_sd   = [0.6 0.6 0.6];       % sacrum TRUE 표준편차 patch (회색)

% 색 조금 밝게 만드는 함수
lighten = @(c) 0.3 + 0.7*c;

% sacrum 모델 누적합 (이미 위에서 M_sum_model_x/z, M_sac_x/z 정의되어 있음)
mu_sum_model_x = nanmean(M_sum_model_x,1);
sd_sum_model_x = nanstd(M_sum_model_x,0,1);
mu_sum_model_z = nanmean(M_sum_model_z,1);
sd_sum_model_z = nanstd(M_sum_model_z,0,1);

% ---------- 0~50% 구간 nRMSE (sacrum model vs TRUE) ----------
idx_half = ph_pct <= 50;  % 0~50% phase 구간 인덱스

% X축
true_sac_x_half_vec  = vec(M_sac_x(:,idx_half));
model_sac_x_half_vec = vec(M_sum_model_x(:,idx_half));
range_sac_x_half = max(true_sac_x_half_vec,[],'omitnan') - min(true_sac_x_half_vec,[],'omitnan');
if range_sac_x_half == 0, range_sac_x_half = 1; end
err_x_half_vec   = model_sac_x_half_vec - true_sac_x_half_vec;
rmse_x_half      = sqrt(nanmean(err_x_half_vec.^2));
nrmse_x_half_pct = (rmse_x_half / range_sac_x_half) * 100;

% Z축
true_sac_z_half_vec  = vec(M_sac_z(:,idx_half));
model_sac_z_half_vec = vec(M_sum_model_z(:,idx_half));
range_sac_z_half = max(true_sac_z_half_vec,[],'omitnan') - min(true_sac_z_half_vec,[],'omitnan');
if range_sac_z_half == 0, range_sac_z_half = 1; end
err_z_half_vec   = model_sac_z_half_vec - true_sac_z_half_vec;
rmse_z_half      = sqrt(nanmean(err_z_half_vec.^2));
nrmse_z_half_pct = (rmse_z_half / range_sac_z_half) * 100;

%% Figure 1) Forward Dynamics — W, WE, ES, SC, Sum, Sacrum TRUE (Mean±SD)
set(0,'DefaultFigureVisible','on');

figure('Name','ACC Mean±SD (All strides/trials)','Units','normalized','Position',[0.05 0.06 0.88 0.82]);
xv = [ph_pct, fliplr(ph_pct)];

% ---------- (1) X 방향 ----------
subplot(2,1,1); hold on; grid on;

% Wrist (W)
p = patch(xv, [mu_wx - sd_wx, fliplr(mu_wx + sd_wx)], lighten(col_wrist), ...
          'EdgeColor','none','FaceAlpha',0.15);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_wx, 'LineWidth',1.8, 'Color',col_wrist, ...
     'DisplayName','Wrist (W)');

% WE segment (increment)
p = patch(xv, [mu_iwx - sd_iwx, fliplr(mu_iwx + sd_iwx)], lighten(col_we), ...
          'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_iwx, 'LineWidth',1.6, 'Color',col_we, ...
     'DisplayName','WE segment');

% ES segment (increment)
p = patch(xv, [mu_esx - sd_esx, fliplr(mu_esx + sd_esx)], lighten(col_es), ...
          'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_esx, 'LineWidth',1.6, 'Color',col_es, ...
     'DisplayName','ES segment');

% SC segment (increment)
p = patch(xv, [mu_scincx - sd_scincx, fliplr(mu_scincx + sd_scincx)], lighten(col_sc), ...
          'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_scincx, 'LineWidth',1.6, 'Color',col_sc, ...
     'DisplayName','SC segment');

% Wrist + WE + ES + SC (sacrum estimate, model)
p = patch(xv, [mu_sum_model_x - sd_sum_model_x, fliplr(mu_sum_model_x + sd_sum_model_x)], ...
          lighten(col_sac_est), 'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_sum_model_x, 'LineWidth',2.0, 'Color',col_sac_est, ...
     'DisplayName','W+WE+ES+SC (model)');

% Sacrum TRUE: patch = 회색, 선 = 진한 검정
if any(isfinite(mu_scx))
    p = patch(xv, [mu_scx - sd_scx, fliplr(mu_scx + sd_scx)], col_sac_sd, ...
              'EdgeColor','none','FaceAlpha',0.18);
    set(p,'HandleVisibility','off');
    plot(ph_pct, mu_scx, 'LineWidth',2.2, 'Color',col_sac_true, ...
         'DisplayName','Sacrum TRUE');
end

xlabel('Stride phase (%)'); ylabel('a_X (m/s^2)');
title(sprintf(['X — Mean±SD [ACC=%s, OFFSET=%s, APPLY_TO_FWD=%d] (N_{used}=%d)\n', ...
               '0–50%% nRMSE (Model vs Sacrum TRUE) = %.2f %%'], ...
      upper(ACC_FRAME), upper(OFFSET_MODE), APPLY_OFFSET_TO_FORWARD, ...
      n_stride_used, nrmse_x_half_pct));
legend('Location','best'); box on;

% ---------- (2) Z 방향 ----------
subplot(2,1,2); hold on; grid on;

% Wrist (W)
p = patch(xv, [mu_wz - sd_wz, fliplr(mu_wz + sd_wz)], lighten(col_wrist), ...
          'EdgeColor','none','FaceAlpha',0.15);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_wz, 'LineWidth',1.8, 'Color',col_wrist, ...
     'DisplayName','Wrist (W)');

% WE segment (increment)
p = patch(xv, [mu_iwz - sd_iwz, fliplr(mu_iwz + sd_iwz)], lighten(col_we), ...
          'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_iwz, 'LineWidth',1.6, 'Color',col_we, ...
     'DisplayName','WE segment');

% ES segment (increment)
p = patch(xv, [mu_esz - sd_esz, fliplr(mu_esz + sd_esz)], lighten(col_es), ...
          'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_esz, 'LineWidth',1.6, 'Color',col_es, ...
     'DisplayName','ES segment');

% SC segment (increment)
p = patch(xv, [mu_scincz - sd_scincz, fliplr(mu_scincz + sd_scincz)], lighten(col_sc), ...
          'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_scincz, 'LineWidth',1.6, 'Color',col_sc, ...
     'DisplayName','SC segment');

% Wrist + WE + ES + SC (sacrum estimate, model)
p = patch(xv, [mu_sum_model_z - sd_sum_model_z, fliplr(mu_sum_model_z + sd_sum_model_z)], ...
          lighten(col_sac_est), 'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_sum_model_z, 'LineWidth',2.0, 'Color',col_sac_est, ...
     'DisplayName','W+WE+ES+SC (model)');

% Sacrum TRUE
if any(isfinite(mu_scz))
    p = patch(xv, [mu_scz - sd_scz, fliplr(mu_scz + sd_scz)], col_sac_sd, ...
              'EdgeColor','none','FaceAlpha',0.18);
    set(p,'HandleVisibility','off');
    plot(ph_pct, mu_scz, 'LineWidth',2.2, 'Color',col_sac_true, ...
         'DisplayName','Sacrum TRUE');
end

xlabel('Stride phase (%)'); ylabel('a_Z (m/s^2)');
title(sprintf(['Z — Mean±SD [ACC=%s, OFFSET=%s, APPLY_TO_FWD=%d] (N_{used}=%d)\n', ...
               '0–50%% nRMSE (Model vs Sacrum TRUE) = %.2f %%'], ...
      upper(ACC_FRAME), upper(OFFSET_MODE), APPLY_OFFSET_TO_FORWARD, ...
      n_stride_used, nrmse_z_half_pct));
legend('Location','best'); box on;

sgtitle('Forward Dynamics — W, WE, ES, SC, Sum, Sacrum TRUE (Mean ± SD)', ...
        'FontSize',14,'FontWeight','bold');


%% Figure 1-1) 모델 비교: 2-Seg vs 3-Seg vs Sacrum TRUE
% - 2-Seg Model: WE+ES (Shoulder 예측) = M_cum
% - 3-Seg Model: WE+ES+SC (Sacrum 예측) = M_cum + M_inc_sc
% - Sacrum TRUE: 실제 Sacrum IMU 측정값 = M_sac

set(0,'DefaultFigureVisible','on');

% 3-Seg 모델 평균/표준편차 계산
M_3seg_x = M_cum_x + M_inc_sc_x;
M_3seg_z = M_cum_z + M_inc_sc_z;
mu_3seg_x = nanmean(M_3seg_x, 1);  sd_3seg_x = nanstd(M_3seg_x, 0, 1);
mu_3seg_z = nanmean(M_3seg_z, 1);  sd_3seg_z = nanstd(M_3seg_z, 0, 1);

% 정확도 계산 (전체 데이터 기준)
vec = @(M) M(:);

% 2-Seg vs Sacrum
corr_2seg_x = corrcoef([vec(M_cum_x), vec(M_sac_x)], 'Rows','pairwise'); r_2seg_x = corr_2seg_x(1,2);
corr_2seg_z = corrcoef([vec(M_cum_z), vec(M_sac_z)], 'Rows','pairwise'); r_2seg_z = corr_2seg_z(1,2);

% 3-Seg vs Sacrum
corr_3seg_x = corrcoef([vec(M_3seg_x), vec(M_sac_x)], 'Rows','pairwise'); r_3seg_x = corr_3seg_x(1,2);
corr_3seg_z = corrcoef([vec(M_3seg_z), vec(M_sac_z)], 'Rows','pairwise'); r_3seg_z = corr_3seg_z(1,2);

figure('Name','Model Comparison: 2-Seg vs 3-Seg vs Sacrum TRUE','Units','normalized','Position',[0.05 0.08 0.90 0.80]);
xv = [ph_pct, fliplr(ph_pct)];

% ==================== X 방향 ====================
subplot(2,1,1); hold on; grid on;

% 2-Seg Model (WE+ES → Shoulder)
p = patch(xv, [mu_cmx - sd_cmx, fliplr(mu_cmx + sd_cmx)], [0.2 0.6 0.9], 'EdgeColor','none','FaceAlpha',0.15);
set(p,'HandleVisibility','off');
h1 = plot(ph_pct, mu_cmx, 'b-', 'LineWidth',2.0, 'DisplayName', ...
    sprintf('2-Seg Model (WE+ES) [r=%.3f]', r_2seg_x));

% 3-Seg Model (WE+ES+SC → Sacrum Pred)
p = patch(xv, [mu_3seg_x - sd_3seg_x, fliplr(mu_3seg_x + sd_3seg_x)], [0.9 0.4 0.2], 'EdgeColor','none','FaceAlpha',0.15);
set(p,'HandleVisibility','off');
h2 = plot(ph_pct, mu_3seg_x, 'r-', 'LineWidth',2.0, 'DisplayName', ...
    sprintf('3-Seg Model (WE+ES+SC) [r=%.3f]', r_3seg_x));

% Sacrum TRUE (IMU 측정값)
p = patch(xv, [mu_scx - sd_scx, fliplr(mu_scx + sd_scx)], [0.2 0.2 0.2], 'EdgeColor','none','FaceAlpha',0.10);
set(p,'HandleVisibility','off');
h3 = plot(ph_pct, mu_scx, 'k--', 'LineWidth',2.5, 'DisplayName','Sacrum TRUE (IMU)');

xlabel('Stride Phase (%)', 'FontSize', 12);
ylabel('Acceleration X (m/s^2)', 'FontSize', 12);
title(sprintf('X Direction — Model Comparison (N_{strides}=%d)', n_stride_with_sacrum), 'FontSize', 13);
legend([h1, h2, h3], 'Location','best', 'FontSize', 10);
box on;
set(gca, 'FontSize', 11);

% ==================== Z 방향 ====================
subplot(2,1,2); hold on; grid on;

% 2-Seg Model (WE+ES → Shoulder)
p = patch(xv, [mu_cmz - sd_cmz, fliplr(mu_cmz + sd_cmz)], [0.2 0.6 0.9], 'EdgeColor','none','FaceAlpha',0.15);
set(p,'HandleVisibility','off');
h1 = plot(ph_pct, mu_cmz, 'b-', 'LineWidth',2.0, 'DisplayName', ...
    sprintf('2-Seg Model (WE+ES) [r=%.3f]', r_2seg_z));

% 3-Seg Model (WE+ES+SC → Sacrum Pred)
p = patch(xv, [mu_3seg_z - sd_3seg_z, fliplr(mu_3seg_z + sd_3seg_z)], [0.9 0.4 0.2], 'EdgeColor','none','FaceAlpha',0.15);
set(p,'HandleVisibility','off');
h2 = plot(ph_pct, mu_3seg_z, 'r-', 'LineWidth',2.0, 'DisplayName', ...
    sprintf('3-Seg Model (WE+ES+SC) [r=%.3f]', r_3seg_z));

% Sacrum TRUE (IMU 측정값)
p = patch(xv, [mu_scz - sd_scz, fliplr(mu_scz + sd_scz)], [0.2 0.2 0.2], 'EdgeColor','none','FaceAlpha',0.10);
set(p,'HandleVisibility','off');
h3 = plot(ph_pct, mu_scz, 'k--', 'LineWidth',2.5, 'DisplayName','Sacrum TRUE (IMU)');

xlabel('Stride Phase (%)', 'FontSize', 12);
ylabel('Acceleration Z (m/s^2)', 'FontSize', 12);
title(sprintf('Z Direction — Model Comparison (N_{strides}=%d)', n_stride_with_sacrum), 'FontSize', 13);
legend([h1, h2, h3], 'Location','best', 'FontSize', 10);
box on;
set(gca, 'FontSize', 11);

sgtitle({'Sacrum Acceleration Estimation: Model vs TRUE', ...
    sprintf('[OFFSET=%s, ES_{scale}=%.3f, SC_{scale}=%.3f]', upper(OFFSET_MODE), ES_SCALE, SC_SCALE)}, ...
    'FontSize', 14, 'FontWeight', 'bold');
%% Figure 2) Relative Acceleration (WE/ES/SC) — TRUE vs MODEL (Mean±SD)
xv = [ph_pct, fliplr(ph_pct)];
col_true_sd = [0.7 0.7 0.7];  % TRUE 표준편차 patch (연한 회색)

figure('Name','Relative Acc (WE/ES/SC) — True vs Model (Mean±SD)', ...
       'Units','normalized','Position',[0.05 0.06 0.88 0.82]);

% WE — X
subplot(3,2,1); hold on; grid on;
% TRUE patch (gray) + line (black)
p = patch(xv, [mu_iwx_t - sd_iwx_t, fliplr(mu_iwx_t + sd_iwx_t)], col_true_sd, ...
          'EdgeColor','none','FaceAlpha',0.15); 
set(p,'HandleVisibility','off');
plot(ph_pct, mu_iwx_t, 'k-', 'LineWidth',1.8, 'DisplayName','WE inc TRUE (X)');
% MODEL patch (light WE color) + line (WE color)
p = patch(xv, [mu_iwx - sd_iwx, fliplr(mu_iwx + sd_iwx)], lighten(col_we), ...
          'EdgeColor','none','FaceAlpha',0.12); 
set(p,'HandleVisibility','off');
plot(ph_pct, mu_iwx, '-', 'LineWidth',1.6, 'Color',col_we, 'DisplayName','WE inc MODEL (X)');
xlabel('Stride phase (%)'); ylabel('\Delta a_X (m/s^2)');
title(sprintf('WE — X   RMSE=%.3f, r=%.3f', rmse_we_x, c_we_x));
legend('Location','best'); box on;
ylim([-8 8])

% WE — Z
subplot(3,2,2); hold on; grid on;
p = patch(xv, [mu_iwz_t - sd_iwz_t, fliplr(mu_iwz_t + sd_iwz_t)], col_true_sd, ...
          'EdgeColor','none','FaceAlpha',0.15);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_iwz_t, 'k-', 'LineWidth',1.8, 'DisplayName','WE inc TRUE (Z)');
p = patch(xv, [mu_iwz - sd_iwz, fliplr(mu_iwz + sd_iwz)], lighten(col_we), ...
          'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_iwz, '-', 'LineWidth',1.6, 'Color',col_we, 'DisplayName','WE inc MODEL (Z)');
xlabel('Stride phase (%)'); ylabel('\Delta a_Z (m/s^2)');
title(sprintf('WE — Z   RMSE=%.3f, r=%.3f', rmse_we_z, c_we_z));
legend('Location','best'); box on;
ylim([-8 8])

% ES — X
subplot(3,2,3); hold on; grid on;
p = patch(xv, [mu_esx_t - sd_esx_t, fliplr(mu_esx_t + sd_esx_t)], col_true_sd, ...
          'EdgeColor','none','FaceAlpha',0.15);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_esx_t, 'k-', 'LineWidth',1.8, 'DisplayName','ES inc TRUE (X)');
p = patch(xv, [mu_esx - sd_esx, fliplr(mu_esx + sd_esx)], lighten(col_es), ...
          'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_esx, '-', 'LineWidth',1.6, 'Color',col_es, 'DisplayName','ES inc MODEL (X)');
xlabel('Stride phase (%)'); ylabel('\Delta a_X (m/s^2)');
title(sprintf('ES — X   RMSE=%.3f, r=%.3f', rmse_es_x, c_es_x));
legend('Location','best'); box on;
ylim([-8 8])

% ES — Z
subplot(3,2,4); hold on; grid on;
p = patch(xv, [mu_esz_t - sd_esz_t, fliplr(mu_esz_t + sd_esz_t)], col_true_sd, ...
          'EdgeColor','none','FaceAlpha',0.15);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_esz_t, 'k-', 'LineWidth',1.8, 'DisplayName','ES inc TRUE (Z)');
p = patch(xv, [mu_esz - sd_esz, fliplr(mu_esz + sd_esz)], lighten(col_es), ...
          'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_esz, '-', 'LineWidth',1.6, 'Color',col_es, 'DisplayName','ES inc MODEL (Z)');
xlabel('Stride phase (%)'); ylabel('\Delta a_Z (m/s^2)');
title(sprintf('ES — Z   RMSE=%.3f, r=%.3f', rmse_es_z, c_es_z));
legend('Location','best'); box on;
ylim([-8 8])

% SC — X
subplot(3,2,5); hold on; grid on;
p = patch(xv, [mu_scx_it - sd_scx_it, fliplr(mu_scx_it + sd_scx_it)], col_true_sd, ...
          'EdgeColor','none','FaceAlpha',0.15);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_scx_it, 'k-', 'LineWidth',1.8, 'DisplayName','SC inc TRUE (X)');
p = patch(xv, [mu_scincx - sd_scincx, fliplr(mu_scincx + sd_scincx)], lighten(col_sc), ...
          'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_scincx, '-', 'LineWidth',1.6, 'Color',col_sc, 'DisplayName','SC inc MODEL (X)');
xlabel('Stride phase (%)'); ylabel('\Delta a_X (m/s^2)');
title(sprintf('SC — X   RMSE=%.3f, r=%.3f', rmse_sc_x, c_sc_x));
legend('Location','best'); box on;
ylim([-8 8])

% SC — Z
subplot(3,2,6); hold on; grid on;
p = patch(xv, [mu_scz_it - sd_scz_it, fliplr(mu_scz_it + sd_scz_it)], col_true_sd, ...
          'EdgeColor','none','FaceAlpha',0.15);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_scz_it, 'k-', 'LineWidth',1.8, 'DisplayName','SC inc TRUE (Z)');
p = patch(xv, [mu_scincz - sd_scincz, fliplr(mu_scincz + sd_scincz)], lighten(col_sc), ...
          'EdgeColor','none','FaceAlpha',0.12);
set(p,'HandleVisibility','off');
plot(ph_pct, mu_scincz, '-', 'LineWidth',1.6, 'Color',col_sc, 'DisplayName','SC inc MODEL (Z)');
xlabel('Stride phase (%)'); ylabel('\Delta a_Z (m/s^2)');
title(sprintf('SC — Z   RMSE=%.3f, r=%.3f', rmse_sc_z, c_sc_z));
legend('Location','best'); box on;

sgtitle(sprintf('Relative Acceleration — TRUE vs MODEL (OFFSET=%s, APPLY_TO_FWD=%d)', ...
        upper(OFFSET_MODE), APPLY_OFFSET_TO_FORWARD), ...
        'FontSize',14,'FontWeight','bold');
ylim([-8 8])

%% Figure 3) 실제 참값의 Sacrum 분해 
if any(isfinite(mu_scx)) || any(isfinite(mu_scz))
    figure('Name','Sacrum Composition (Cumulative Mean±SD)','Units','normalized','Position',[0.06 0.08 0.86 0.70]);
    xv = [ph_pct, fliplr(ph_pct)];

    % --- X 방향 ---
    subplot(2,1,1); hold on; grid on;
    p = patch(xv, [mu_wtx - sd_wtx, fliplr(mu_wtx + sd_wtx)], [0.6 0.6 0.6], 'EdgeColor','none','FaceAlpha',0.08); set(p,'HandleVisibility','off');
    plot(ph_pct, mu_wtx,'Color', [1 1 0], 'LineWidth',1.6, 'DisplayName','wrist TRUE (X)');

    p = patch(xv, [(mu_wtx+mu_iwx_t) - sqrt(sd_wtx.^2+sd_iwx_t.^2), ...
                   fliplr((mu_wtx+mu_iwx_t) + sqrt(sd_wtx.^2+sd_iwx_t.^2))], [0.85 0.4 0.4], 'EdgeColor','none','FaceAlpha',0.08); set(p,'HandleVisibility','off');
    plot(ph_pct, mu_wtx+mu_iwx_t, 'r-', 'LineWidth',1.6, 'DisplayName','wrist + WE (X)');

    p = patch(xv, [(mu_wtx+mu_iwx_t+mu_esx_t) - sqrt(sd_wtx.^2+sd_iwx_t.^2+sd_esx_t.^2), ...
                   fliplr((mu_wtx+mu_iwx_t+mu_esx_t) + sqrt(sd_wtx.^2+sd_iwx_t.^2+sd_esx_t.^2))], [0.4 0.6 0.9], 'EdgeColor','none','FaceAlpha',0.08); set(p,'HandleVisibility','off');
    plot(ph_pct, mu_wtx+mu_iwx_t+mu_esx_t, 'b-', 'LineWidth',1.6, 'DisplayName','wrist + WE + ES (X)');

    p = patch(xv, [mu_sumx - sd_sumx, fliplr(mu_sumx + sd_sumx)], [0.2 0.8 0.5], 'EdgeColor','none','FaceAlpha',0.08); set(p,'HandleVisibility','off');
    plot(ph_pct, mu_sumx, 'g-', 'LineWidth',1.8, 'DisplayName','wrist + WE + ES + SC (X)');

    p = patch(xv, [mu_scx - sd_scx, fliplr(mu_scx + sd_scx)], [0.1 0.1 0.1], 'EdgeColor','none','FaceAlpha',0.05); set(p,'HandleVisibility','off');
    plot(ph_pct, mu_scx, 'k--', 'LineWidth',2.0, 'DisplayName','sacrum TRUE (X)');

    xlabel('Stride phase (%)'); ylabel('a_X (m/s^2)');
    title('Sacrum Composition (X)');
    legend('Location','best'); box on;

    % --- Z 방향 ---
    subplot(2,1,2); hold on; grid on;
    p = patch(xv, [mu_wtz - sd_wtz, fliplr(mu_wtz + sd_wtz)], [0.6 0.6 0.6], 'EdgeColor','none','FaceAlpha',0.08); set(p,'HandleVisibility','off');
    plot(ph_pct, mu_wtz,'Color', [1 1 0], 'LineWidth',1.6, 'DisplayName','wrist TRUE (Z)');

    p = patch(xv, [(mu_wtz+mu_iwz_t) - sqrt(sd_wtz.^2+sd_iwz_t.^2), ...
                   fliplr((mu_wtz+mu_iwz_t) + sqrt(sd_wtz.^2+sd_iwz_t.^2))], [0.85 0.4 0.4], 'EdgeColor','none','FaceAlpha',0.08); set(p,'HandleVisibility','off');
    plot(ph_pct, mu_wtz+mu_iwz_t, 'r-', 'LineWidth',1.6, 'DisplayName','wrist + WE (Z)');

    p = patch(xv, [(mu_wtz+mu_iwz_t+mu_esz_t) - sqrt(sd_wtz.^2+sd_iwz_t.^2+sd_esz_t.^2), ...
                   fliplr((mu_wtz+mu_iwz_t+mu_esz_t) + sqrt(sd_wtz.^2+sd_iwz_t.^2+sd_esz_t.^2))], [0.4 0.6 0.9], 'EdgeColor','none','FaceAlpha',0.08); set(p,'HandleVisibility','off');
    plot(ph_pct, mu_wtz+mu_iwz_t+mu_esz_t, 'b-', 'LineWidth',1.6, 'DisplayName','wrist + WE + ES (Z)');

    p = patch(xv, [mu_sumz - sd_sumz, fliplr(mu_sumz + sd_sumz)], [0.2 0.8 0.5], 'EdgeColor','none','FaceAlpha',0.08); set(p,'HandleVisibility','off');
    plot(ph_pct, mu_sumz, 'g-', 'LineWidth',1.8, 'DisplayName','wrist + WE + ES + SC (Z)');

    p = patch(xv, [mu_scz - sd_scz, fliplr(mu_scz + sd_scz)], [0.1 0.1 0.1], 'EdgeColor','none','FaceAlpha',0.05); set(p,'HandleVisibility','off');
    plot(ph_pct, mu_scz, 'k--', 'LineWidth',2.0, 'DisplayName','sacrum TRUE (Z)');

    xlabel('Stride phase (%)'); ylabel('a_Z (m/s^2)');
    title('Sacrum Composition (Z)');
    legend('Location','best'); box on;

    sgtitle('Sacrum Acceleration Decomposition (Cumulative build-up)','FontSize',14,'FontWeight','bold');
else
    warning('Sacrum IMU가 없어 누적 figure는 건너뜁니다.');
end
