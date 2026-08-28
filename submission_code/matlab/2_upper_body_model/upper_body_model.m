% ================================================================
%  Per-stride modelling -> forward dynamics (WE -> ES [-> SC]) -> writes stride_modeling
%  - Iterates over every subject / session / day / stride
%  - Wrist acceleration source: WRIST_SRC = 'global' | 'local' | 'none'
%  - Segment scaling: ES_SCALE, SC_SCALE (applied to theta, omega and alpha)
%  - A constant angular offset is estimated and applied for each stride
%  - OFFSET_MODE = 'auto' | 'off' | 'manual' | 'gravity'
%      * gravity: uses vWE.offset_index / offset_rad
%                 off_we = offset_rad - th_we_pred(offset_index)
%                 off_es = es_offset_scaling_factor * off_we
%                 off_sc = sc_offset_scaling_factor * off_we
%      * auto   : circular-mean offset against the measured angle (WE and ES separately); SC has no measured angle, so a scaling factor is used
%  - With INCLUDE_SC = true the chain is extended to SC and stored under 'sacrum'
% ================================================================
clc; % close all;

% ---------------- user options ----------------
fs  = 100;         % Hz
dt  = 1/fs;

% selection filters (empty = process everything)
target_subjects = {};    % e.g. {'S001','S002'}
target_sessions = {};    % e.g. {'ss1','ss2'}
target_days     = {};    % e.g. {'Day1','Day3'}
stride_range    = [];    % e.g. [3 10]

% wrist acceleration source: 'global' | 'local' | 'none'
WRIST_SRC = 'none';

% segment scaling
ES_SCALE = 0.497;
SC_SCALE = 0.12;       % SC = SC_SCALE * WE

% whether to apply the angular offset in the forward pass
APPLY_OFFSET_TO_FORWARD = true;

% === offset mode ===
OFFSET_MODE   = 'gravity';          % 'auto' | 'off' | 'manual' | 'gravity'
MANUAL_OFF_WE = deg2rad(118.25);   % rad
MANUAL_OFF_ES = deg2rad(88.10);    % rad
MANUAL_OFF_SC = deg2rad(0.0);      % rad

% offset scaling for ES / SC (SC in gravity and auto modes, ES in gravity mode)
es_offset_scaling_factor = 0.76;
sc_offset_scaling_factor = -1.2;

% include the SC (torso) segment
INCLUDE_SC = false;

% ---------------- input checks ----------------
if ~(exist('stride_kinematics_arm','var')==1 && exist('stride_kinematics_arm_vector','var')==1)
    error('stride_kinematics_arm and stride_kinematics_arm_vector are required.');
end
if ~isfield(stride_kinematics_arm,'L_Wrist')
    error('stride_kinematics_arm.L_Wrist is missing.');
end

% ---------------- output struct ----------------
stride_modeling = struct();
% stride_modeling.options = struct( ...
%     'fs',fs, 'WRIST_SRC',WRIST_SRC, ...
%     'ES_SCALE',ES_SCALE, 'SC_SCALE',SC_SCALE, ...
%     'APPLY_OFFSET_TO_FORWARD',APPLY_OFFSET_TO_FORWARD, ...
%     'OFFSET_MODE',OFFSET_MODE, ...
%     'MANUAL_OFF_WE',MANUAL_OFF_WE,'MANUAL_OFF_ES',MANUAL_OFF_ES,'MANUAL_OFF_SC',MANUAL_OFF_SC, ...
%     'es_offset_scaling_factor',es_offset_scaling_factor, ...
%     'sc_offset_scaling_factor',sc_offset_scaling_factor, ...
%     'INCLUDE_SC',INCLUDE_SC);

% ---------------- helpers ----------------
circ_offset = @(trueAng, predAng) atan2(nanmean(sin(trueAng - predAng)), ...
                                        nanmean(cos(trueAng - predAng)));

% ---------------- loop: subject / session / day ----------------
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

            % ---- leafs ----
            if ~isfield(stride_kinematics_arm.L_Wrist.(subj).(sess), day), continue; end
            wLeaf = stride_kinematics_arm.L_Wrist.(subj).(sess).(day);

            if ~isfield(stride_kinematics_arm.L_Elbow.(subj).(sess), day), continue; end
            eLeaf = stride_kinematics_arm.L_Elbow.(subj).(sess).(day);

            if ~isfield(stride_kinematics_arm.L_Shoulder.(subj).(sess), day), continue; end
            sLeaf = stride_kinematics_arm.L_Shoulder.(subj).(sess).(day);

            if isfield(stride_kinematics_arm,'sacrum') && ...
               isfield(stride_kinematics_arm.sacrum, subj) && ...
               isfield(stride_kinematics_arm.sacrum.(subj), sess) && ...
               isfield(stride_kinematics_arm.sacrum.(subj).(sess), day)
                cLeaf = stride_kinematics_arm.sacrum.(subj).(sess).(day);
            else
                cLeaf = [];
            end

            % ---- vectors (WE/ES[/SC]) ----
            okV = true;
            if isfield(stride_kinematics_arm_vector,'L_Wrist_to_L_Elbow') && ...
               isfield(stride_kinematics_arm_vector.L_Wrist_to_L_Elbow, subj) && ...
               isfield(stride_kinematics_arm_vector.L_Wrist_to_L_Elbow.(subj), sess) && ...
               isfield(stride_kinematics_arm_vector.L_Wrist_to_L_Elbow.(subj).(sess), day)
                vWE = stride_kinematics_arm_vector.L_Wrist_to_L_Elbow.(subj).(sess).(day);
            else, okV = false; end

            if isfield(stride_kinematics_arm_vector,'L_Elbow_to_L_Shoulder') && ...
               isfield(stride_kinematics_arm_vector.L_Elbow_to_L_Shoulder, subj) && ...
               isfield(stride_kinematics_arm_vector.L_Elbow_to_L_Shoulder.(subj), sess) && ...
               isfield(stride_kinematics_arm_vector.L_Elbow_to_L_Shoulder.(subj).(sess), day)
                vES = stride_kinematics_arm_vector.L_Elbow_to_L_Shoulder.(subj).(sess).(day);
            else, okV = false; end

            if INCLUDE_SC
                if isfield(stride_kinematics_arm_vector,'L_Shoulder_to_sacrum') && ...
                   isfield(stride_kinematics_arm_vector.L_Shoulder_to_sacrum, subj) && ...
                   isfield(stride_kinematics_arm_vector.L_Shoulder_to_sacrum.(subj), sess) && ...
                   isfield(stride_kinematics_arm_vector.L_Shoulder_to_sacrum.(subj).(sess), day)
                    vSC = stride_kinematics_arm_vector.L_Shoulder_to_sacrum.(subj).(sess).(day);
                else, okV = false; end
            end
            if ~okV, warning('missing vector node: %s.%s.%s', subj,sess,day); continue; end

            % number of strides
            if ~isfield(wLeaf,'imu_gyro_local_y') || ~iscell(wLeaf.imu_gyro_local_y) || isempty(wLeaf.imu_gyro_local_y)
                warning('imu_gyro_local_y missing or empty: L_Wrist.%s.%s.%s', subj,sess,day); continue;
            end
            nStride = numel(wLeaf.imu_gyro_local_y);

            % preallocate output cells
            for jn = ["L_Wrist","L_Elbow","L_Shoulder","sacrum"]
                stride_modeling.(jn).(subj).(sess).(day).acc_x = cell(nStride,1);
                stride_modeling.(jn).(subj).(sess).(day).acc_z = cell(nStride,1);
            end

            % stride range
            if isempty(stride_range)
                s1=1; s2=nStride;
            else
                s1 = max(1, stride_range(1));
                s2 = min(nStride, stride_range(2));
                if s1> s2, warning('invalid stride_range: %s-%s %s', subj, sess, day); continue; end
            end

            % ================= STRIDE LOOP =================
            for s = s1:s2
                okField = @(leaf,fn) isfield(leaf,fn) && iscell(leaf.(fn)) && numel(leaf.(fn))>=s && ~isempty(leaf.(fn){s});
                if ~(okField(wLeaf,'pos_x') && okField(wLeaf,'pos_z') && ...
                     okField(eLeaf,'pos_x') && okField(eLeaf,'pos_z') && ...
                     okField(sLeaf,'pos_x') && okField(sLeaf,'pos_z') && ...
                     okField(wLeaf,'imu_gyro_local_y'))
                    continue;
                end

                wX = wLeaf.pos_x{s}(:);  wZ = wLeaf.pos_z{s}(:);
                eX = eLeaf.pos_x{s}(:);  eZ = eLeaf.pos_z{s}(:);
                sX = sLeaf.pos_x{s}(:);  sZ = sLeaf.pos_z{s}(:);
                gy = wLeaf.imu_gyro_local_y{s}(:);

                % Wrist acc
                switch lower(WRIST_SRC)
                    case 'global'
                        if okField(wLeaf,'imu_acc_global_x') && okField(wLeaf,'imu_acc_global_z')
                            ax_w = wLeaf.imu_acc_global_x{s}(:);
                            az_w = wLeaf.imu_acc_global_z{s}(:);
                        else, warning('wrist global acc missing: %s-%s %s stride %d', subj,sess,day,s); continue; end
                    case 'local'
                        if okField(wLeaf,'imu_acc_local_x') && okField(wLeaf,'imu_acc_local_z')
                            ax_w = wLeaf.imu_acc_local_x{s}(:);
                            az_w = wLeaf.imu_acc_local_z{s}(:);
                        else, warning('wrist local acc missing: %s-%s %s stride %d', subj,sess,day,s); continue; end
                    case 'none'
                        ax_w = []; az_w = [];
                    otherwise, error('invalid WRIST_SRC: %s', WRIST_SRC);
                end

                % Lengths
                getLen = @(vLeaf) (isfield(vLeaf,'length_xz') && isfield(vLeaf.length_xz,'pos') && ...
                                   iscell(vLeaf.length_xz.pos) && numel(vLeaf.length_xz.pos)>=s && ~isempty(vLeaf.length_xz.pos{s}));
                if ~getLen(vWE) || ~getLen(vES), continue; end
                L_we = mean(vWE.length_xz.pos{s}(:),'omitnan');
                L_es = mean(vES.length_xz.pos{s}(:),'omitnan');
                if INCLUDE_SC
                    if ~getLen(vSC), continue; end
                    L_sc = mean(vSC.length_xz.pos{s}(:),'omitnan');
                end

                % common length
                Ls = [numel(wX), numel(wZ), numel(eX), numel(eZ), numel(sX), numel(sZ), numel(gy)];
                if ~isempty(ax_w), Ls(end+1)=numel(ax_w); Ls(end+1)=numel(az_w); end
                N = min(Ls);
                if N < 5, continue; end

                wX=wX(1:N); wZ=wZ(1:N); eX=eX(1:N); eZ=eZ(1:N); sX=sX(1:N); sZ=sZ(1:N);
                gy=gy(1:N);
                if isempty(ax_w), ax_w=zeros(N,1); az_w=zeros(N,1);
                else, ax_w=ax_w(1:N); az_w=az_w(1:N); end
                t=(0:N-1)'/fs;

                % gyro sign and offset correction
                idxF=isfinite(gy);
                if any(idxF), gy=gy-median(gy(idxF)); end
                if any(~idxF)
                    if nnz(idxF)>=2, gy(~idxF)=interp1(t(idxF),gy(idxF),t(~idxF),'linear','extrap');
                    else, gy(~idxF)=0; end
                end

                % predicted angles
                th_we_pred = cumtrapz(t, gy); th_we_pred = th_we_pred - th_we_pred(1);
                th_es_pred = ES_SCALE * th_we_pred;
                if INCLUDE_SC, th_sc_pred = SC_SCALE * th_we_pred; end

                % measured angles
                th_we_true = atan2(eZ - wZ, eX - wX);
                th_es_true = atan2(sZ - eZ, sX - eX);

                % ----- offset mode -----
                switch lower(OFFSET_MODE)
                    case 'off'
                        off_we = 0.0; off_es = 0.0; off_sc = 0.0;

                    case 'manual'
                        off_we = MANUAL_OFF_WE;
                        off_es = MANUAL_OFF_ES;
                        off_sc = MANUAL_OFF_SC;

                    case 'auto'
                        % offset estimated against the measured angle
                        off_we = circ_offset(th_we_true, th_we_pred);
                        off_es = circ_offset(th_es_true, th_es_pred);
                        % SC has no measured angle, so a scaling factor is used (set to 0 to disable)
                        off_sc = sc_offset_scaling_factor * off_we;

                    case 'gravity'
                        % uses vWE.offset_index / vWE.offset_rad
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
                            off_we = atan2( sin(target_rad - th_we_pred(k0)), cos(target_rad - th_we_pred(k0)) );
                        else
                            % fallback: auto (measured-angle based)
                            off_we = circ_offset(th_we_true, th_we_pred);
                        end
                        off_es = es_offset_scaling_factor * off_we;
                        off_sc = sc_offset_scaling_factor * off_we;

                    otherwise
                        warning('unknown OFFSET_MODE=%s, falling back to auto', OFFSET_MODE);
                        off_we = circ_offset(th_we_true, th_we_pred);
                        off_es = circ_offset(th_es_true, th_es_pred);
                        off_sc = sc_offset_scaling_factor * off_we;
                end

                % angles used in the forward pass
                if APPLY_OFFSET_TO_FORWARD
                    th_we_fd = th_we_pred + off_we;
                    th_es_fd = th_es_pred + off_es;
                    if INCLUDE_SC, th_sc_fd = th_sc_pred + off_sc; end
                else
                    th_we_fd = th_we_pred;
                    th_es_fd = th_es_pred;
                    if INCLUDE_SC, th_sc_fd = th_sc_pred; end
                end

                % angular velocity and acceleration
                om_we = gy;               al_we = gradient(om_we, dt);
                om_es = ES_SCALE*om_we;   al_es = ES_SCALE*al_we;
                if INCLUDE_SC
                    om_sc = SC_SCALE*om_we; al_sc = SC_SCALE*al_we;
                end

                % 2-D relative acceleration terms (Eq. 3)
                rddx_we = -L_we .* ( cos(th_we_fd).*(om_we.^2) + sin(th_we_fd).*al_we );
                rddz_we = -L_we .* ( sin(th_we_fd).*(om_we.^2) - cos(th_we_fd).*al_we );

                rddx_es = -L_es .* ( cos(th_es_fd).*(om_es.^2) + sin(th_es_fd).*al_es );
                rddz_es = -L_es .* ( sin(th_es_fd).*(om_es.^2) - cos(th_es_fd).*al_es );

                if INCLUDE_SC
                    rddx_sc = -L_sc .* ( cos(th_sc_fd).*(om_sc.^2) + sin(th_sc_fd).*al_sc );
                    rddz_sc = -L_sc .* ( sin(th_sc_fd).*(om_sc.^2) - cos(th_sc_fd).*al_sc );
                end

                % chain accumulation (Eq. 3-4)
                ax_elbow    = ax_w + rddx_we;
                az_elbow    = az_w + rddz_we;

                ax_shoulder = ax_elbow + rddx_es;
                az_shoulder = az_elbow + rddz_es;

                if INCLUDE_SC
                    ax_sacrum_pred = ax_shoulder + rddx_sc;
                    az_sacrum_pred = az_shoulder + rddz_sc;
                end

                % store
                stride_modeling.L_Wrist.(subj).(sess).(day).acc_x{s,1}    = ax_w(:);
                stride_modeling.L_Wrist.(subj).(sess).(day).acc_z{s,1}    = az_w(:);

                stride_modeling.L_Elbow.(subj).(sess).(day).acc_x{s,1}    = ax_elbow(:);
                stride_modeling.L_Elbow.(subj).(sess).(day).acc_z{s,1}    = az_elbow(:);

                stride_modeling.L_Shoulder.(subj).(sess).(day).acc_x{s,1} = ax_shoulder(:);
                stride_modeling.L_Shoulder.(subj).(sess).(day).acc_z{s,1} = az_shoulder(:);

                if INCLUDE_SC
                    stride_modeling.sacrum.(subj).(sess).(day).acc_x{s,1} = ax_sacrum_pred(:);
                    stride_modeling.sacrum.(subj).(sess).(day).acc_z{s,1} = az_sacrum_pred(:);
                else
                    stride_modeling.sacrum.(subj).(sess).(day).acc_x{s,1} = ax_shoulder(:);
                    stride_modeling.sacrum.(subj).(sess).(day).acc_z{s,1} = az_shoulder(:);
                end

                % store offset metadata
                % stride_modeling.meta.(subj).(sess).(day).off_we{s,1} = off_we;
                % stride_modeling.meta.(subj).(sess).(day).off_es{s,1} = off_es;
                if INCLUDE_SC
                    stride_modeling.meta.(subj).(sess).(day).off_sc{s,1} = off_sc;
                end
            end
        end
    end
end

disp('stride_modeling created.');
save('stride_modeling','stride_modeling')
% clearvars -except kinematics_arm grf stride_kinematics_arm stride_grf kinematics_arm_vector stride_kinematics_arm_vector stride_modeling
