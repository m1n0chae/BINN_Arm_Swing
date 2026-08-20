%% Section 0) Data 준비 
% load('grf.mat')
% load('kinematics_arm.mat')
% load('kinematics_arm_vector.mat')
load('valid_ranges_gyro.mat')  % valid_ranges 변수가 있어야 합니다.

%% Section 1) Gyro(imu_gyro_local_y) 기준으로 stride 제작 + 표 생성 (GRF는 비교용 집계만)
clc;
% ---- 파라미터 ----
fc_lp_gyro   = 2;         % gyro 저역통과(Hz)  ← 2 Hz
fc_lp_grf    = 20;        % Fz 저역통과(Hz) (비교용)
minStrideSec = 0.70;      % (참고용) 이벤트 간 최소 간격(초) - 현재 Gyro에는 사용 안 함
thr_frac_grf = 0.20;      % GRF 비교용 임계값 = thr_frac_grf * max(Fz_filtered)
MIN_DIST_FRAMES_GYRO = 70; % Gyro 이벤트 최소 간격(프레임) 고정

stride_grf                   = struct();
stride_kinematics_arm        = struct();
stride_kinematics_arm_vector = struct();
segmentation_tables          = struct();   % 표 저장용 컨테이너
summary_rows                 = {};         % 전체 요약 표를 만들기 위한 로우 버퍼

% (필수 변수 확인)
if ~exist('grf','var'), error('workspace에 grf가 필요합니다.'); end
if ~exist('kinematics_arm','var'), error('workspace에 kinematics_arm이 필요합니다.'); end
if ~exist('kinematics_arm_vector','var'), error('workspace에 kinematics_arm_vector가 필요합니다.'); end
if ~exist('valid_ranges','var')
    error('workspace에 valid_ranges가 필요합니다. 범위 선택 스크립트를 먼저 실행하세요.');
end

axes3    = {'Fx','Fy','Fz'};
subjects = fieldnames(grf);

for iS = 1:numel(subjects)
    S = subjects{iS};
    if ~isstruct(grf.(S)), continue; end

    sessions = fieldnames(grf.(S));
    for iSS = 1:numel(sessions)
        SS = sessions{iSS};
        if ~isstruct(grf.(S).(SS)), continue; end

        days = fieldnames(grf.(S).(SS));
        for iD = 1:numel(days)
            DY = days{iD};
            if ~isstruct(grf.(S).(SS).(DY)), continue; end

            % 필수 데이터 존재 확인
            if ~(isfield(grf.(S).(SS).(DY),'Left') && isfield(grf.(S).(SS).(DY).Left,'Fz'))
                warning('Left Fz 없음 (스킵): grf.%s.%s.%s.Left.Fz', S,SS,DY);
                continue;
            end

            % 연속 time과 fs 추정 (선호: kinematics time → grf.time → fallback=100Hz)
            FzL = grf.(S).(SS).(DY).Left.Fz;
            if isrow(FzL), FzL = FzL.'; end
            Nref = numel(FzL);
            if Nref == 0
                warning('FzL 데이터가 비어있음 (스킵): %s/%s/%s', S,SS,DY);
                continue;
            end

            fs_here = NaN; tvec = [];

            % 1) kinematics time 탐색
            segNames_for_time = fieldnames(kinematics_arm);
            for ig = 1:numel(segNames_for_time)
                seg = segNames_for_time{ig};
                okPath = isfield(kinematics_arm,seg) && isfield(kinematics_arm.(seg),S) ...
                      && isfield(kinematics_arm.(seg).(S),SS) && isfield(kinematics_arm.(seg).(S).(SS),DY);
                if ~okPath, continue; end
                nodeA = kinematics_arm.(seg).(S).(SS).(DY);
                if isfield(nodeA,'time') && isnumeric(nodeA.time) && numel(nodeA.time)==Nref
                    tvec = nodeA.time(:);
                    dt = median(diff(tvec(~isnan(tvec))));
                    if isfinite(dt) && dt>0, fs_here = 1/dt; end
                    break;
                end
            end

            % 2) grf.time 사용
            if isempty(tvec)
                if isfield(grf.(S).(SS).(DY),'time') && numel(grf.(S).(SS).(DY).time)==Nref
                    tvec = grf.(S).(SS).(DY).time(:);
                    dt = median(diff(tvec(~isnan(tvec))));
                    if isfinite(dt) && dt>0, fs_here = 1/dt; end
                end
            end

            % 3) fallback
            if isnan(fs_here), fs_here = 100; end
            if isempty(tvec),  tvec    = (0:Nref-1)'/fs_here; end

            % 연속 gyro 신호 확보 (L_Wrist imu_gyro_local_y)
            if ~(isfield(kinematics_arm,'L_Wrist') && isfield(kinematics_arm.L_Wrist,S) && ...
                 isfield(kinematics_arm.L_Wrist.(S),SS) && isfield(kinematics_arm.L_Wrist.(S).(SS),DY) && ...
                 isfield(kinematics_arm.L_Wrist.(S).(SS).(DY),'imu_gyro_local_y'))
                warning('연속 gyro 신호 없음(스킵): kinematics_arm.L_Wrist.%s.%s.%s.imu_gyro_local_y',S,SS,DY);
                continue;
            end
            gy_full = kinematics_arm.L_Wrist.(S).(SS).(DY).imu_gyro_local_y;
            gy_full = gy_full(:);
            if numel(gy_full) ~= Nref
                warning('gyro 길이 불일치(N=%d vs Nref=%d)로 스킵: %s/%s/%s',numel(gy_full),Nref,S,SS,DY);
                continue;
            end

            % 필터링: gyro(2Hz), grf(Fz)
            gyf = double(gy_full);
            if fc_lp_gyro > 0 && fc_lp_gyro < fs_here/2
                [b1,a1] = butter(4, fc_lp_gyro/(fs_here/2), 'low');
                gyf = filtfilt(b1,a1,gyf);
            end

            FL = double(FzL);
            if fc_lp_grf > 0 && fc_lp_grf < fs_here/2
                [b2,a2] = butter(4, fc_lp_grf/(fs_here/2), 'low');
                FLf = filtfilt(b2,a2,FL);
            else
                FLf = FL;
            end

            % 비교용 GRF 이벤트(상승 엣지)
            thrL   = thr_frac_grf * max(FLf);
            aboveL = FLf > thrL;
            HS_L_grf = find( aboveL(2:end) & ~aboveL(1:end-1) ) + 1;  % GRF 기준 지점

            % Gyro 최소 피크(국소 최소) 검출: 70프레임 간격 고정
            invgy     = -gyf;  % local minima 검출용
            rngG      = max(invgy) - min(invgy);
            promA     = 0.7*std(gyf);
            promB     = 0.05*rngG;
            minProm   = max([promA, promB, eps]);

            [~, locsG] = findpeaks(invgy, ...
                                    'MinPeakDistance', MIN_DIST_FRAMES_GYRO, ...
                                    'MinPeakProminence', minProm);

            HS_L_gyro = locsG(:);
            if numel(HS_L_gyro) < 2
                warning('Gyro 기준 stride가 부족(이벤트<2): %s/%s/%s', S,SS,DY);
                continue;
            end

            % Gyro 기반 stride 인덱스
            idxG    = [HS_L_gyro(1:end-1) HS_L_gyro(2:end)-1];
            nStride = size(idxG,1);

            % valid_ranges 기반 stride 필터링 (gyro 구간에 대해)
            if ~(isfield(valid_ranges,S) && isfield(valid_ranges.(S),SS) && isfield(valid_ranges.(S).(SS),DY))
                warning('valid_ranges에 %s/%s/%s 항목이 없습니다. 이 Trial의 모든 Stride를 스킵합니다.', S,SS,DY);
                idxG = [];
            else
                current_valid_ranges = valid_ranges.(S).(SS).(DY);
                if isempty(current_valid_ranges)
                    fprintf('[필터링] %s/%s/%s | 유효 범위가 없어 %d개 stride 모두 제거\n', S,SS,DY, nStride);
                    idxG = [];
                else
                    keep_mask = false(nStride, 1);
                    for kk = 1:nStride
                        i1 = idxG(kk, 1);
                        i2 = idxG(kk, 2);
                        stride_t_start = tvec(i1);
                        stride_t_end   = tvec(i2);

                        is_valid_stride = false;
                        for m = 1:size(current_valid_ranges, 1)
                            range_start = current_valid_ranges(m, 1);
                            range_end   = current_valid_ranges(m, 2);
                            if stride_t_start >= range_start && stride_t_end <= range_end
                                is_valid_stride = true; break;
                            end
                        end
                        keep_mask(kk) = is_valid_stride;
                    end
                    n_removed = nStride - sum(keep_mask);
                    if n_removed > 0
                        fprintf('[필터링] %s/%s/%s | %d개 중 %d개 제거\n', S,SS,DY, nStride, n_removed);
                    end
                    idxG = idxG(keep_mask, :);
                end
            end

            nStride = size(idxG, 1);
            if nStride == 0
                fprintf('[결과] %s/%s/%s | 유효한 gyro 기반 stride 없음\n', S,SS,DY);
                continue;
            end

            % 저장 컨테이너 보장
            if ~isfield(stride_grf,S), stride_grf.(S)=struct(); end
            if ~isfield(stride_grf.(S),SS), stride_grf.(S).(SS)=struct(); end
            if ~isfield(stride_grf.(S).(SS),DY), stride_grf.(S).(SS).(DY)=struct(); end
            if ~isfield(stride_grf.(S).(SS).(DY),'Left'),  stride_grf.(S).(SS).(DY).Left  = struct(); end
            if ~isfield(stride_grf.(S).(SS).(DY),'Right'), stride_grf.(S).(SS).(DY).Right = struct(); end
            if ~isfield(stride_grf.(S).(SS).(DY),'Total'), stride_grf.(S).(SS).(DY).Total = struct(); end
            if ~isfield(segmentation_tables,S), segmentation_tables.(S)=struct(); end
            if ~isfield(segmentation_tables.(S),SS), segmentation_tables.(S).(SS)=struct(); end

            % 길이·정합성 사전 점검 (GRF는 slicing만 바뀜: gy 기준 경계 사용)
            mismatch_found = false;
            mismatch_msgs  = {};
            for ia = 1:numel(axes3)
                ax = axes3{ia};
                Lsig_chk = grf.(S).(SS).(DY).Left.(ax);  if isrow(Lsig_chk),  Lsig_chk  = Lsig_chk.';  end
                Rsig_chk = grf.(S).(SS).(DY).Right.(ax); if isrow(Rsig_chk), Rsig_chk = Rsig_chk.'; end

                if numel(Lsig_chk) ~= Nref
                    mismatch_found = true;
                    mismatch_msgs{end+1} = sprintf('Left.%s=%d (Nref=%d)', ax, numel(Lsig_chk), Nref);
                end
                if numel(Rsig_chk) ~= Nref
                    mismatch_found = true;
                    mismatch_msgs{end+1} = sprintf('Right.%s=%d (Nref=%d)', ax, numel(Rsig_chk), Nref);
                end
            end

            if mismatch_found
                fprintf('[stride SKIP | GRF 길이 불일치] %s/%s/%s\n', S,SS,DY);
                for mm = 1:numel(mismatch_msgs)
                    fprintf('    - %s\n', mismatch_msgs{mm});
                end
                % time 길이 로그
                len_grf_time = 0;
                if isfield(grf.(S).(SS).(DY),'time') && isnumeric(grf.(S).(SS).(DY).time)
                    len_grf_time = numel(grf.(S).(SS).(DY).time);
                end
                len_mocap_time = 0;
                for ig_len2 = 1:numel(segNames_for_time)
                    seg_len2 = segNames_for_time{ig_len2};
                    okPath_len2 = isfield(kinematics_arm,seg_len2) && isfield(kinematics_arm.(seg_len2),S) ...
                               && isfield(kinematics_arm.(seg_len2).(S),SS) && isfield(kinematics_arm.(seg_len2).(S).(SS),DY);
                    if ~okPath_len2, continue; end
                    node_len2 = kinematics_arm.(seg_len2).(S).(SS).(DY);
                    if isfield(node_len2,'time') && isnumeric(node_len2.time) && ~isempty(node_len2.time)
                        len_mocap_time = numel(node_len2.time(:));
                        break;
                    end
                end
                len_imu_time = 0;
                if isfield(kinematics_arm,'L_Wrist') && isfield(kinematics_arm.L_Wrist,S) && ...
                   isfield(kinematics_arm.L_Wrist.(S),SS) && isfield(kinematics_arm.L_Wrist.(S).(SS),DY) && ...
                   isfield(kinematics_arm.L_Wrist.(S).(SS).(DY),'imu_time') && ...
                   isnumeric(kinematics_arm.L_Wrist.(S).(SS).(DY).imu_time)
                    len_imu_time = numel(kinematics_arm.L_Wrist.(S).(SS).(DY).imu_time);
                end
                fprintf('    참고 | grf.time=%d | mocap.time=%d | imu_time=%d\n', len_grf_time, len_mocap_time, len_imu_time);
                continue;
            end

            % Gyro 경계로 GRF 잘라 저장 (Left/Right/Total, Fx/Fy/Fz)
            for ia = 1:numel(axes3)
                ax = axes3{ia};
                Lsig = grf.(S).(SS).(DY).Left.(ax);  if isrow(Lsig),  Lsig  = Lsig.';  end
                Rsig = grf.(S).(SS).(DY).Right.(ax); if isrow(Rsig), Rsig = Rsig.'; end
                Tsig = Lsig + Rsig;

                CL = cell(nStride,1); CR = cell(nStride,1); CT = cell(nStride,1);
                for kk = 1:nStride
                    i1 = idxG(kk,1); i2 = idxG(kk,2);
                    if i2 <= i1, error('인덱스 이상(%s): %s/%s/%s', ax, S,SS,DY); end
                    CL{kk} = Lsig(i1:i2);
                    CR{kk} = Rsig(i1:i2);
                    CT{kk} = Tsig(i1:i2);
                end
                stride_grf.(S).(SS).(DY).Left.(ax)  = CL;
                stride_grf.(S).(SS).(DY).Right.(ax) = CR;
                stride_grf.(S).(SS).(DY).Total.(ax) = CT;
            end
            stride_grf.(S).(SS).(DY).counts = struct('left',nStride,'right',nStride,'total',nStride);

            % kinematics_arm 잘라 저장 (Nref와 길이가 같은 축만 gyro 경계로)
            segNamesA = fieldnames(kinematics_arm);
            for ig = 1:numel(segNamesA)
                seg = segNamesA{ig};
                okPath = isfield(kinematics_arm,seg) && isfield(kinematics_arm.(seg),S) ...
                      && isfield(kinematics_arm.(seg).(S),SS) && isfield(kinematics_arm.(seg).(S).(SS),DY);
                if ~okPath, continue; end
                nodeA = kinematics_arm.(seg).(S).(SS).(DY);
                outA  = struct();
                fA = fieldnames(nodeA);
                for ia2 = 1:numel(fA)
                    fld = fA{ia2};
                    val = nodeA.(fld);
                    if isnumeric(val)
                        sz = size(val);
                        dims_eq = find(sz == Nref);
                        if ~isempty(dims_eq)
                            d = dims_eq(1);
                            nd = ndims(val);
                            C = cell(nStride,1);
                            for kk = 1:nStride
                                i1 = idxG(kk,1); i2 = idxG(kk,2);
                                subs = repmat({':'},1,nd); subs{d}=i1:i2; 
                                C{kk}=val(subs{:});
                            end
                            outA.(fld) = C;
                        else
                            outA.(fld) = val;
                        end
                    else
                        outA.(fld) = val;
                    end
                end
                if ~isfield(stride_kinematics_arm,seg), stride_kinematics_arm.(seg)=struct(); end
                if ~isfield(stride_kinematics_arm.(seg),S),  stride_kinematics_arm.(seg).(S)=struct(); end
                if ~isfield(stride_kinematics_arm.(seg).(S),SS), stride_kinematics_arm.(seg).(S).(SS)=struct(); end
                stride_kinematics_arm.(seg).(S).(SS).(DY) = outA;
            end

            % kinematics_arm_vector 잘라 저장 (gyro 경계로)
            pairNames = fieldnames(kinematics_arm_vector);
            for ip = 1:numel(pairNames)
                pair = pairNames{ip};
                okPair = isfield(kinematics_arm_vector.(pair),S) ...
                      && isfield(kinematics_arm_vector.(pair).(S),SS) ...
                      && isfield(kinematics_arm_vector.(pair).(S).(SS),DY);
                if ~okPair, continue; end

                nodeV = kinematics_arm_vector.(pair).(S).(SS).(DY);
                outV  = struct();
                fV1 = fieldnames(nodeV);
                for iv1 = 1:numel(fV1)
                    fld1 = fV1{iv1};
                    val1 = nodeV.(fld1);

                    if isnumeric(val1)
                        sz1 = size(val1);
                        dims_eq1 = find(sz1 == Nref);
                        if ~isempty(dims_eq1)
                            d1 = dims_eq1(1);
                            nd1 = ndims(val1);
                            C1 = cell(nStride,1);
                            for kk = 1:nStride
                                i1 = idxG(kk,1); i2 = idxG(kk,2);
                                subs = repmat({':'},1,nd1); subs{d1}=i1:i2; 
                                C1{kk}=val1(subs{:});
                            end
                            outV.(fld1) = C1;
                        else
                            outV.(fld1) = val1;
                        end

                    elseif isstruct(val1)
                        sub = struct(); 
                        fV2 = fieldnames(val1);
                        for iv2 = 1:numel(fV2)
                            fld2 = fV2{iv2}; 
                            val2 = val1.(fld2);
                            if isnumeric(val2)
                                sz2 = size(val2); 
                                dims_eq2 = find(sz2 == Nref);
                                if ~isempty(dims_eq2)
                                    d2 = dims_eq2(1); nd2 = ndims(val2);
                                    C2 = cell(nStride,1);
                                    for kk = 1:nStride
                                        i1 = idxG(kk,1); i2 = idxG(kk,2);
                                        subs = repmat({':'},1,nd2); subs{d2}=i1:i2; 
                                        C2{kk}=val2(subs{:});
                                    end
                                    sub.(fld2) = C2;
                                else
                                    sub.(fld2) = val2;
                                end
                            else
                                sub.(fld2) = val2;
                            end
                        end
                        outV.(fld1) = sub;

                    else
                        outV.(fld1) = val1;
                    end
                end

                if ~isfield(stride_kinematics_arm_vector,pair), stride_kinematics_arm_vector.(pair)=struct(); end
                if ~isfield(stride_kinematics_arm_vector.(pair),S),  stride_kinematics_arm_vector.(pair).(S)=struct(); end
                if ~isfield(stride_kinematics_arm_vector.(pair).(S),SS)
                    stride_kinematics_arm_vector.(pair).(S).(SS)=struct();
                end
                stride_kinematics_arm_vector.(pair).(S).(SS).(DY) = outV;
            end

            % ===== 표 생성 및 표시 =====
            gyro_event_idx   = HS_L_gyro(:);
            gyro_event_time  = tvec(HS_L_gyro(:));
            grf_event_idx    = HS_L_grf(:);
            grf_event_time   = tvec(HS_L_grf(:));

            % 이벤트 표
            T_events_gyro = table( (1:numel(gyro_event_idx))', gyro_event_idx, gyro_event_time, ...
                                   'VariableNames', {'event_id','event_index','time_s'} );

            T_events_grf  = table( (1:numel(grf_event_idx))', grf_event_idx, grf_event_time, ...
                                   'VariableNames', {'event_id','event_index','time_s'} );

            % 스트라이드 표(gyro 기준)
            stride_id      = (1:nStride)';
            start_idx      = idxG(:,1);
            end_idx        = idxG(:,2);
            start_time_s   = tvec(start_idx);
            end_time_s     = tvec(end_idx);
            duration_s     = end_time_s - start_time_s;

            T_strides_gyro = table(stride_id, start_idx, end_idx, start_time_s, end_time_s, duration_s, ...
                                   'VariableNames', {'stride_id','start_idx','end_idx','start_time_s','end_time_s','duration_s'});

            % 저장
            if ~isfield(segmentation_tables.(S).(SS),DY), segmentation_tables.(S).(SS).(DY)=struct(); end
            segmentation_tables.(S).(SS).(DY).events_gyro   = T_events_gyro;
            segmentation_tables.(S).(SS).(DY).events_grf    = T_events_grf;
            segmentation_tables.(S).(SS).(DY).strides_gyro  = T_strides_gyro;
            segmentation_tables.(S).(SS).(DY).n_gyro        = size(T_strides_gyro,1);
            segmentation_tables.(S).(SS).(DY).n_grf         = max(0, size(T_events_grf,1) - 1); % GRF 이벤트가 k개면 stride는 k-1

            % 콘솔 표시
            fprintf('\n================ %s / %s / %s ================\n', S,SS,DY);
            fprintf('총 스트라이드(gyro 기준): %d\n', segmentation_tables.(S).(SS).(DY).n_gyro);
            if ~isempty(T_events_gyro)
                fprintf('자른 시점(gyro 이벤트, 초):\n');
                disp(T_events_gyro);
            end
            if ~isempty(T_strides_gyro)
                fprintf('스트라이드 구간 표(gyro 기준):\n');
                disp(T_strides_gyro);
            end

            % 전체 요약용 로우 추가
            meanDur = mean(duration_s,'omitnan');
            summary_rows(end+1,:) = {S, SS, DY, segmentation_tables.(S).(SS).(DY).n_gyro, segmentation_tables.(S).(SS).(DY).n_grf, meanDur}; %#ok<AGROW>

            fprintf('[done-gyro] %s/%s/%s | 유효 strides: %d (gyro 기준)\n', S,SS,DY, nStride);
        end
    end
end

% 전체 요약 표
if ~isempty(summary_rows)
    summary_all = cell2table(summary_rows, ...
        'VariableNames', {'Subject','Session','Day','N_Gyro_Strides','N_GRF_Strides','MeanStrideDur_s'});
    fprintf('\n=========== 전체 요약 ===========\n');
    disp(summary_all);
else
    warning('요약을 만들 데이터가 없습니다.');
end

%% Section 2) Resample all stride cells to 101 points + per-stride real time (gyro 기준)
targetN = 101;
xi = linspace(0,1,targetN);
axes3  = {'Fx','Fy','Fz'};
sides3 = {'Left','Right','Total'};

if ~exist('stride_grf','var'), error('stride_grf가 필요합니다.'); end

subjects = fieldnames(stride_grf);
for iS = 1:numel(subjects)
    S  = subjects{iS};
    if ~isstruct(stride_grf.(S)), continue; end

    sessions = fieldnames(stride_grf.(S));
    for iSS = 1:numel(sessions)
        SS = sessions{iSS};
        if ~isstruct(stride_grf.(S).(SS)), continue; end

        days = fieldnames(stride_grf.(S).(SS));
        for iD = 1:numel(days)
            DY = days{iD};
            if ~isstruct(stride_grf.(S).(SS).(DY)), continue; end

            % fs_here 재추정
            fs_here = NaN;
            segNames = fieldnames(kinematics_arm);
            for ig = 1:numel(segNames)
                seg = segNames{ig};
                if  isfield(kinematics_arm.(seg),S) && ...
                    isfield(kinematics_arm.(seg).(S),SS) && ...
                    isfield(kinematics_arm.(seg).(S).(SS),DY) && ...
                    isfield(kinematics_arm.(seg).(S).(SS).(DY),'time')
                    tv = kinematics_arm.(seg).(S).(SS).(DY).time; 
                    if isnumeric(tv) && numel(tv) > 1
                        tv = tv(:); 
                        dt = median(diff(tv(~isnan(tv))));
                        if isfinite(dt) && dt > 0, fs_here = 1/dt; end
                        break;
                    end
                end
            end
            if isnan(fs_here)
                if isfield(grf,S) && isfield(grf.(S),SS) && isfield(grf.(S).(SS),DY) && isfield(grf.(S).(SS).(DY),'time')
                    tv = grf.(S).(SS).(DY).time;
                    if isnumeric(tv) && numel(tv) > 1
                        tv = tv(:); 
                        dt = median(diff(tv(~isnan(tv))));
                        if isfinite(dt) && dt > 0, fs_here = 1/dt; end
                    end
                end
            end
            if isnan(fs_here), fs_here = 100; end

            if ~isfield(stride_grf.(S).(SS).(DY),'Left') || ~isfield(stride_grf.(S).(SS).(DY).Left,'Fz')
                continue;
            end
            nStride = numel(stride_grf.(S).(SS).(DY).Left.Fz);
            if nStride == 0, continue; end

            % per-stride 실제 시간축 (원 길이와 fs 기반)
            time_cells = cell(nStride,1);
            L_Fz = stride_grf.(S).(SS).(DY).Left.Fz;
            for k = 1:nStride
                y = L_Fz{k};
                if isempty(y), warning('빈 stride 발견: %s/%s/%s (k=%d)',S,SS,DY,k); continue; end
                n0 = numel(y);
                dur = (n0-1) / fs_here;
                time_cells{k} = linspace(0, dur, targetN).';
            end

            % GRF 보간
            for isd = 1:numel(sides3)
                sd = sides3{isd};
                if ~isfield(stride_grf.(S).(SS).(DY), sd), continue; end
                for iax = 1:numel(axes3)
                    ax = axes3{iax};
                    if ~isfield(stride_grf.(S).(SS).(DY).(sd), ax), continue; end
                    C = stride_grf.(S).(SS).(DY).(sd).(ax);
                    if ~iscell(C), continue; end
                    for k = 1:numel(C)
                        y = C{k};
                        if isempty(y), continue; end
                        y = double(y(:).');
                        x = linspace(0,1,numel(y));
                        C{k} = interp1(x, y, xi, 'pchip').';
                    end
                    stride_grf.(S).(SS).(DY).(sd).(ax) = C;
                end
                stride_grf.(S).(SS).(DY).(sd).time = time_cells;
            end

            % kinematics_arm 보간
            segNames2 = fieldnames(stride_kinematics_arm);
            for ig = 1:numel(segNames2)
                seg = segNames2{ig};
                if ~isfield(stride_kinematics_arm.(seg),S) || ...
                   ~isfield(stride_kinematics_arm.(seg).(S),SS) || ...
                   ~isfield(stride_kinematics_arm.(seg).(S).(SS),DY)
                    continue;
                end
                nodeA = stride_kinematics_arm.(seg).(S).(SS).(DY);
                fA = fieldnames(nodeA);
                for ia = 1:numel(fA)
                    fld = fA{ia};
                    val = nodeA.(fld);
                    if iscell(val)
                        for k = 1:numel(val)
                            Y = val{k};
                            if isempty(Y) || ~isnumeric(Y), continue; end
                            Y = double(Y);
                            sz = size(Y);
                            nd = ndims(Y);
                            [~, d] = max(sz);
                            if isempty(d), d = 1; end
                            perm = 1:nd; perm([d nd]) = [nd d];
                            Yp = permute(Y, perm);
                            szp = size(Yp);
                            m = prod(szp(1:end-1)); 
                            n = szp(end);
                            if n <= 1, val{k} = Y; continue; end
                            X = linspace(0,1,n);
                            M = reshape(Yp, [m n]);
                            M2 = zeros(m, targetN);
                            for r = 1:m
                                M2(r,:) = interp1(X, M(r,:), xi, 'pchip');
                            end
                            Yp2 = reshape(M2, [szp(1:end-1) targetN]);
                            val{k} = ipermute(Yp2, perm);
                        end
                        nodeA.(fld) = val;
                    end
                end
                nodeA.time = time_cells;
                stride_kinematics_arm.(seg).(S).(SS).(DY) = nodeA;
            end

            % kinematics_arm_vector 보간
            pairNames = fieldnames(stride_kinematics_arm_vector);
            for ip = 1:numel(pairNames)
                pair = pairNames{ip};
                if ~isfield(stride_kinematics_arm_vector.(pair),S) || ...
                   ~isfield(stride_kinematics_arm_vector.(pair).(S),SS) || ...
                   ~isfield(stride_kinematics_arm_vector.(pair).(S).(SS),DY)
                    continue;
                end
                nodeV = stride_kinematics_arm_vector.(pair).(S).(SS).(DY);
                fV1 = fieldnames(nodeV);
                for iv1 = 1:numel(fV1)
                    fld1 = fV1{iv1};
                    val1 = nodeV.(fld1);
                    if iscell(val1)
                        for k = 1:numel(val1)
                            Y = val1{k};
                            if isempty(Y) || ~isnumeric(Y), continue; end
                            Y = double(Y);
                            sz = size(Y); nd = ndims(Y);
                            [~, d] = max(sz); if isempty(d), d = 1; end
                            perm = 1:nd; perm([d nd]) = [nd d];
                            Yp = permute(Y, perm);
                            szp = size(Yp); m = prod(szp(1:end-1)); n = szp(end);
                            if n <= 1, val1{k} = Y; continue; end
                            X = linspace(0,1,n);
                            M = reshape(Yp, [m n]); 
                            M2 = zeros(m, targetN);
                            for r = 1:m
                                M2(r,:) = interp1(X, M(r,:), xi, 'pchip');
                            end
                            Yp2 = reshape(M2, [szp(1:end-1) targetN]);
                            val1{k} = ipermute(Yp2, perm);
                        end
                        nodeV.(fld1) = val1;

                    elseif isstruct(val1)
                        fV2 = fieldnames(val1);
                        for iv2 = 1:numel(fV2)
                            fld2 = fV2{iv2}; 
                            val2 = val1.(fld2);
                            if ~iscell(val2), continue; end
                            for k = 1:numel(val2)
                                Y = val2{k};
                                if isempty(Y) || ~isnumeric(Y), continue; end
                                Y = double(Y);
                                sz = size(Y); nd = ndims(Y);
                                [~, d] = max(sz); if isempty(d), d = 1; end
                                perm = 1:nd; perm([d nd]) = [nd d];
                                Yp = permute(Y, perm);
                                szp = size(Yp); m = prod(szp(1:end-1)); n = szp(end);
                                if n <= 1, val2{k} = Y; continue; end
                                X = linspace(0,1,n);
                                M = reshape(Yp, [m n]); 
                                M2 = zeros(m, targetN);
                                for r = 1:m
                                    M2(r,:) = interp1(X, M(r,:), xi, 'pchip');
                                end
                                Yp2 = reshape(M2, [szp(1:end-1) targetN]);
                                val2{k} = ipermute(Yp2, perm);
                            end
                            val1.(fld2) = val2;
                        end
                        nodeV.(fld1) = val1;
                    end
                end
                nodeV.time = time_cells;
                stride_kinematics_arm_vector.(pair).(S).(SS).(DY) = nodeV;
            end

            fprintf('[resampled] %s/%s/%s | stride cells to %d pts + time added (gyro 기준)\n', S,SS,DY,targetN);
        end
    end
end

%% Section 3) 데이터 저장 
% clearvars -except marker_for_imu_orientation kinematics_arm kinematics_arm_vector grf stride_grf stride_kinematics_arm stride_kinematics_arm_vector valid_ranges segmentation_tables summary_all
save('stride_grf_gyro.mat','stride_grf')
save('stride_kinematics_arm_gyro.mat','stride_kinematics_arm')
save('stride_kinematics_arm_vector_gyro.mat','stride_kinematics_arm_vector')
% save('segmentation_tables.mat','segmentation_tables','summary_all')
disp('필터링된 Stride 데이터 저장 완료. (gyro 기준)')

%% ===== Summary: Gyro vs GRF stride counts AFTER valid_ranges filtering =====
clc;
fc_lp_gyro   = 2;          % gyro 저역통과(Hz)
fc_lp_grf    = 20;         % Fz 저역통과(Hz)
thr_frac_grf = 0.10;       % GRF 임계값 비율
MIN_DIST_FRAMES_GYRO = 70; % Gyro 이벤트 최소 간격(프레임)

subjects = fieldnames(grf);

C_Subject  = {};
C_Session  = {};
C_Day      = {};
C_NGyV     = [];
C_NGfV     = [];
C_MeanGy   = [];
C_MeanGf   = [];
C_VApplied = [];

for iS = 1:numel(subjects)
    S = subjects{iS}; if ~isstruct(grf.(S)), continue; end
    sessions = fieldnames(grf.(S));
    for iSS = 1:numel(sessions)
        SS = sessions{iSS}; if ~isstruct(grf.(S).(SS)), continue; end
        days = fieldnames(grf.(S).(SS));
        for iD = 1:numel(days)
            DY = days{iD}; if ~isstruct(grf.(S).(SS).(DY)), continue; end
            if ~(isfield(grf.(S).(SS).(DY),'Left') && isfield(grf.(S).(SS).(DY).Left,'Fz')), continue; end

            % --- 시간축 및 fs ---
            FzL = grf.(S).(SS).(DY).Left.Fz; if isempty(FzL), continue; end
            if isrow(FzL), FzL = FzL.'; end
            Nref = numel(FzL);
            tvec = []; fs_here = NaN;

            % kinematics time 우선
            segNames_for_time = fieldnames(kinematics_arm);
            for ig = 1:numel(segNames_for_time)
                seg = segNames_for_time{ig};
                okPath = isfield(kinematics_arm,seg) && isfield(kinematics_arm.(seg),S) && ...
                         isfield(kinematics_arm.(seg).(S),SS) && isfield(kinematics_arm.(seg).(S).(SS),DY);
                if ~okPath, continue; end
                nodeA = kinematics_arm.(seg).(S).(SS).(DY);
                if isfield(nodeA,'time') && isnumeric(nodeA.time) && numel(nodeA.time)==Nref
                    tvec = nodeA.time(:);
                    dt = median(diff(tvec(~isnan(tvec))));
                    if isfinite(dt) && dt>0, fs_here = 1/dt; end
                    break;
                end
            end
            % grf.time 대안
            if isempty(tvec) && isfield(grf.(S).(SS).(DY),'time') && ...
               isnumeric(grf.(S).(SS).(DY).time) && numel(grf.(S).(SS).(DY).time)==Nref
                tvec = grf.(S).(SS).(DY).time(:);
                dt = median(diff(tvec(~isnan(tvec))));
                if isfinite(dt) && dt>0, fs_here = 1/dt; end
            end
            if isempty(tvec),  tvec = (0:Nref-1)'/100; end
            if isnan(fs_here), fs_here = 1/median(diff(tvec)); end

            % --- 신호 준비 ---
            FL = double(FzL);
            if fc_lp_grf > 0 && fc_lp_grf < fs_here/2
                [b2,a2] = butter(4, fc_lp_grf/(fs_here/2), 'low');
                FLf = filtfilt(b2,a2,FL);
            else
                FLf = FL;
            end
            thrL   = thr_frac_grf * max(FLf);
            aboveL = FLf > thrL;
            HS_L   = find( aboveL(2:end) & ~aboveL(1:end-1) ) + 1;

            % GRF stride
            idxGf = [];
            if numel(HS_L) >= 2
                idxGf = [HS_L(1:end-1) HS_L(2:end)-1];
            end

            % Gyro 준비
            hasGy = isfield(kinematics_arm,'L_Wrist') && isfield(kinematics_arm.L_Wrist,S) && ...
                    isfield(kinematics_arm.L_Wrist.(S),SS) && isfield(kinematics_arm.L_Wrist.(S).(SS),DY) && ...
                    isfield(kinematics_arm.L_Wrist.(S).(SS).(DY),'imu_gyro_local_y');
            idxGy = [];
            if hasGy
                gy_full = kinematics_arm.L_Wrist.(S).(SS).(DY).imu_gyro_local_y(:);
                if numel(gy_full)==Nref
                    gyf = double(gy_full);
                    if fc_lp_gyro > 0 && fc_lp_gyro < fs_here/2
                        [b1,a1] = butter(4, fc_lp_gyro/(fs_here/2), 'low');
                        gyf = filtfilt(b1,a1,gyf);
                    end
                    invgy   = -gyf;
                    rngG    = max(invgy) - min(invgy);
                    promA   = 0.7*std(gyf);
                    promB   = 0.05*rngG;
                    minProm = max([promA, promB, eps]);
                    [~, locsG] = findpeaks(invgy, ...
                                           'MinPeakDistance', MIN_DIST_FRAMES_GYRO, ...
                                           'MinPeakProminence', minProm);
                    if numel(locsG) >= 2
                        idxGy = [locsG(1:end-1) locsG(2:end)-1];
                    end
                end
            end

            % --- valid_ranges 적용(완전 포함) ---
            have_valid = exist('valid_ranges','var')==1 && ...
                         isfield(valid_ranges,S) && isfield(valid_ranges.(S),SS) && ...
                         isfield(valid_ranges.(S).(SS),DY) && ~isempty(valid_ranges.(S).(SS).(DY));
            if have_valid
                ranges = valid_ranges.(S).(SS).(DY);
            else
                ranges = [];
            end

            % Gyro after valid
            nGyro_valid = 0; meanGy_valid = NaN;
            if ~isempty(idxGy)
                keep_mask_gy = true(size(idxGy,1),1);
                if ~isempty(ranges)
                    keep_mask_gy = false(size(idxGy,1),1);
                    for k = 1:size(idxGy,1)
                        t1 = tvec(idxGy(k,1)); t2 = tvec(idxGy(k,2));
                        kept = false;
                        for m = 1:size(ranges,1)
                            if (t1 >= ranges(m,1)) && (t2 <= ranges(m,2)), kept = true; break; end
                        end
                        keep_mask_gy(k) = kept;
                    end
                end
                idxGyV = idxGy(keep_mask_gy,:);
                nGyro_valid = size(idxGyV,1);
                if nGyro_valid>0
                    dur = tvec(idxGyV(:,2)) - tvec(idxGyV(:,1));
                    meanGy_valid = mean(dur,'omitnan');
                end
            end

            % GRF after valid
            nGrf_valid = 0; meanGf_valid = NaN;
            if ~isempty(idxGf)
                keep_mask_gf = true(size(idxGf,1),1);
                if ~isempty(ranges)
                    keep_mask_gf = false(size(idxGf,1),1);
                    for k = 1:size(idxGf,1)
                        t1 = tvec(idxGf(k,1)); t2 = tvec(idxGf(k,2));
                        kept = false;
                        for m = 1:size(ranges,1)
                            if (t1 >= ranges(m,1)) && (t2 <= ranges(m,2)), kept = true; break; end
                        end
                        keep_mask_gf(k) = kept;
                    end
                end
                idxGfV = idxGf(keep_mask_gf,:);
                nGrf_valid = size(idxGfV,1);
                if nGrf_valid>0
                    dur = tvec(idxGfV(:,2)) - tvec(idxGfV(:,1));
                    meanGf_valid = mean(dur,'omitnan');
                end
            end

            % --- 결과 누적 ---
            C_Subject{end+1,1}  = S;
            C_Session{end+1,1}  = SS;
            C_Day{end+1,1}      = DY;
            C_NGyV(end+1,1)     = nGyro_valid;
            C_NGfV(end+1,1)     = nGrf_valid;
            C_MeanGy(end+1,1)   = meanGy_valid;
            C_MeanGf(end+1,1)   = meanGf_valid;
            C_VApplied(end+1,1) = logical(~isempty(ranges));
        end
    end
end

T_summary = table( ...
    C_Subject, C_Session, C_Day, ...
    C_NGyV, C_NGfV, C_MeanGy, C_MeanGf, C_VApplied, ...
    'VariableNames', {'Subject','Session','Day', ...
                      'N_Gyro_Strides_Valid','N_GRF_Strides_Valid', ...
                      'MeanStrideDur_s_Gyro','MeanStrideDur_s_GRF','ValidApplied'});

if ~isempty(T_summary)
    if ~isstring(T_summary.Subject), T_summary.Subject = string(T_summary.Subject); end
    if ~isstring(T_summary.Session), T_summary.Session = string(T_summary.Session); end
    if ~isstring(T_summary.Day),     T_summary.Day     = string(T_summary.Day);     end
    T_summary = sortrows(T_summary, {'Subject','Session','Day'});
end

disp(T_summary);

%% ===== Plot only mismatched trials (|N_Gyro_Strides_Valid - N_GRF_Strides_Valid| >= 2) =====
clc;
fc_lp_gyro   = 2;          % gyro 저역통과(Hz)
fc_lp_grf    = 20;         % Fz 저역통과(Hz)
thr_frac_grf = 0.10;       % GRF 임계값 비율
MIN_DIST_FRAMES_GYRO = 70; % Gyro 이벤트 최소 간격(프레임)

if ~isstring(T_summary.Subject), T_summary.Subject = string(T_summary.Subject); end
if ~isstring(T_summary.Session), T_summary.Session = string(T_summary.Session); end
if ~isstring(T_summary.Day),     T_summary.Day     = string(T_summary.Day);     end

diff_cnt = abs(T_summary.N_Gyro_Strides_Valid - T_summary.N_GRF_Strides_Valid);
idx_mis  = find(diff_cnt >= 2);

fprintf('[INFO] 불일치 트라이얼 수: %d\n', numel(idx_mis));
if isempty(idx_mis)
    disp('모든 트라이얼에서 유효 스트라이드 개수 차이가 2 미만입니다.');
end

for ii = 1:numel(idx_mis)
    r  = idx_mis(ii);
    S  = char(T_summary.Subject(r));
    SS = char(T_summary.Session(r));
    DY = char(T_summary.Day(r));

    ok_grf  = isfield(grf,S) && isfield(grf.(S),SS) && isfield(grf.(S).(SS),DY) && ...
              isfield(grf.(S).(SS).(DY),'Left') && isfield(grf.(S).(SS).(DY).Left,'Fz');
    ok_gyro = isfield(kinematics_arm,'L_Wrist') && isfield(kinematics_arm.L_Wrist,S) && ...
              isfield(kinematics_arm.L_Wrist.(S),SS) && isfield(kinematics_arm.L_Wrist.(S).(SS),DY) && ...
              isfield(kinematics_arm.L_Wrist.(S).(SS).(DY),'imu_gyro_local_y');
    if ~(ok_grf && ok_gyro)
        fprintf('[SKIP] 데이터 없음: %s / %s / %s\n', S,SS,DY); continue;
    end

    FzL = grf.(S).(SS).(DY).Left.Fz; if isrow(FzL), FzL = FzL.'; end
    gy  = kinematics_arm.L_Wrist.(S).(SS).(DY).imu_gyro_local_y(:);
    Nref = numel(FzL);

    % 시간축 및 fs
    tvec = []; fs_here = NaN;
    segNames_for_time = fieldnames(kinematics_arm);
    for ig = 1:numel(segNames_for_time)
        seg = segNames_for_time{ig};
        okPath = isfield(kinematics_arm,seg) && isfield(kinematics_arm.(seg),S) && ...
                 isfield(kinematics_arm.(seg).(S),SS) && isfield(kinematics_arm.(seg).(S).(SS),DY);
        if ~okPath, continue; end
        nodeA = kinematics_arm.(seg).(S).(SS).(DY);
        if isfield(nodeA,'time') && isnumeric(nodeA.time) && numel(nodeA.time)==Nref
            tvec = nodeA.time(:);
            dt = median(diff(tvec(~isnan(tvec)))); if isfinite(dt) && dt>0, fs_here = 1/dt; end
            break;
        end
    end
    if isempty(tvec) && isfield(grf.(S).(SS).(DY),'time') && ...
       isnumeric(grf.(S).(SS).(DY).time) && numel(grf.(S).(SS).(DY).time)==Nref
        tvec = grf.(S).(SS).(DY).time(:);
        dt = median(diff(tvec(~isnan(tvec)))); if isfinite(dt) && dt>0, fs_here = 1/dt; end
    end
    if isempty(tvec),  tvec = (0:Nref-1)'/100; end
    if isnan(fs_here), fs_here = 1/median(diff(tvec)); end

    % 길이 일치화
    if numel(gy) ~= Nref
        nmin = min(numel(gy), Nref);
        FzL = FzL(1:nmin); gy = gy(1:nmin); tvec = tvec(1:nmin); Nref = nmin;
        fprintf('[WARN] 길이 불일치로 절단: %s/%s/%s\n', S,SS,DY);
    end

    % 필터링
    gyf = double(gy);
    if fc_lp_gyro > 0 && fc_lp_gyro < fs_here/2
        [b1,a1] = butter(4, fc_lp_gyro/(fs_here/2), 'low');
        gyf = filtfilt(b1,a1,gyf);
    end
    FL = double(FzL);
    if fc_lp_grf > 0 && fc_lp_grf < fs_here/2
        [b2,a2] = butter(4, fc_lp_grf/(fs_here/2), 'low');
        FLf = filtfilt(b2,a2,FL);
    else
        FLf = FL;
    end

    % 이벤트 검출
    thrL   = thr_frac_grf * max(FLf);
    aboveL = FLf > thrL;
    HS_L   = find( aboveL(2:end) & ~aboveL(1:end-1) ) + 1;

    invgy   = -gyf;
    rngG    = max(invgy) - min(invgy);
    promA   = 0.7*std(gyf);
    promB   = 0.05*rngG;
    minProm = max([promA, promB, eps]);
    [~, locsG] = findpeaks(invgy, ...
                           'MinPeakDistance', MIN_DIST_FRAMES_GYRO, ...
                           'MinPeakProminence', minProm);

    % valid range 패치용
    have_valid = exist('valid_ranges','var')==1 && ...
                 isfield(valid_ranges,S) && isfield(valid_ranges.(S),SS) && ...
                 isfield(valid_ranges.(S).(SS),DY) && ~isempty(valid_ranges.(S).(SS).(DY));
    if have_valid
        ranges = valid_ranges.(S).(SS).(DY);
    else
        ranges = [];
    end

    % 그래프
    figure('Name', sprintf('%s / %s / %s  |  mismatch=%d', ...
           S,SS,DY, abs(T_summary.N_Gyro_Strides_Valid(r)-T_summary.N_GRF_Strides_Valid(r))), ...
           'Units','normalized','Position',[0.08 0.08 0.84 0.78], 'Color','w');

    % Gyro
    subplot(2,1,1); hold on; grid on; box on;
    plot(tvec, gyf, 'k-', 'LineWidth',1.2, 'DisplayName','Gyro local y (LPF 2 Hz)');
    yl1 = ylim;
    for m = 1:size(ranges,1)
        xs = [ranges(m,1) ranges(m,2)];
        patch([xs(1) xs(2) xs(2) xs(1)], [yl1(1) yl1(1) yl1(2) yl1(2)], [0.6 0.9 0.6], ...
              'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
    end
    for k = 1:numel(locsG)
        xline(tvec(locsG(k)), ':', 'Color',[0.2 0.6 0.9], 'LineWidth',1.0, 'HandleVisibility','off');
    end
    plot(tvec, gyf, 'k-', 'LineWidth',1.2, 'HandleVisibility','off');
    title(sprintf('Gyro  |  N_{valid}(Gyro)=%d, N_{valid}(GRF)=%d', ...
          T_summary.N_Gyro_Strides_Valid(r), T_summary.N_GRF_Strides_Valid(r)));
    xlabel('time (s)'); ylabel('gyro (deg/s or rad/s)');
    legend('Location','best');

    % GRF
    subplot(2,1,2); hold on; grid on; box on;
    plot(tvec, FLf, '-', 'LineWidth',1.2, 'DisplayName','GRF Left Fz (LPF 20 Hz)');
    yl2 = ylim;
    yThr = min(max(thrL, yl2(1)), yl2(2));
    plot(tvec, ones(size(tvec))*yThr, '--', 'LineWidth',1.0, 'DisplayName','GRF threshold');
    for m = 1:size(ranges,1)
        xs = [ranges(m,1) ranges(m,2)];
        patch([xs(1) xs(2) xs(2) xs(1)], [yl2(1) yl2(1) yl2(2) yl2(2)], [0.6 0.9 0.6], ...
              'FaceAlpha',0.10,'EdgeColor','none','HandleVisibility','off');
    end
    for k = 1:numel(HS_L)
        xline(tvec(HS_L(k)), '-', 'Color',[0.85 0.2 0.2], 'LineWidth',0.8, 'HandleVisibility','off');
    end
    plot(tvec, FLf, '-', 'LineWidth',1.2, 'HandleVisibility','off');
    title('GRF Left Fz with gyro and GRF events');
    xlabel('time (s)'); ylabel('Fz (N)');
    legend('Location','best');

    fprintf('[PLOT] %3d/%3d  %s / %s / %s  |  Gyro=%d, GRF=%d, Diff=%d\n', ...
        ii, numel(idx_mis), S,SS,DY, ...
        T_summary.N_Gyro_Strides_Valid(r), T_summary.N_GRF_Strides_Valid(r), ...
        abs(T_summary.N_Gyro_Strides_Valid(r)-T_summary.N_GRF_Strides_Valid(r)));
    drawnow;
end

