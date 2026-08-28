%% Section 0) Data load & Table (Updated: Flex/Ext + Ipsi/Contra + Peak/Integral)
% =========================================================
%  1) 최상위 폴더 설정 및 파일 탐색
% =========================================================
% clear; clc; close all;
% [사용자 설정] 최상위 데이터 폴더 경로
%   eval_day1_on_full.py / eval_percent_models_on_full.py 가 쓴 OUTPUT_ROOT 를 가리킨다.
%   그 아래에 모델별로 Figure3_loso / Figure4_speedloso / Figure5_datasize /
%   Figure6_datasize_sploso 폴더와 per-stride 예측 CSV 가 들어 있어야 한다.
target_root_folder = "../../../eval_outputs";
% [사용자 설정] 사용할 stride 인덱스 ( [] = 전체 )
stride_idx_list = [];
% -------------------------------------------------
% [Phase Definition]Data
% -------------------------------------------------
idx_flex = 20:70;               % Flexion
idx_ext  = [71:101, 1:19];      % Extension
idx_ipsi   = 1:41;              % Ipsilateral (0~40%)
idx_contra = 51:91;             % Contralateral (50~90%)
% -------------------------------------------------
% 1. 재귀적으로 모든 CSV 파일 검색
% -------------------------------------------------
fprintf('폴더 검색 중: %s ...\n', target_root_folder);
file_list = dir(fullfile(target_root_folder, '**', '*.csv'));
valid_idx = ~startsWith({file_list.name}, '._');
file_list = file_list(valid_idx);
n_files = length(file_list);
if n_files < 1
    error('CSV 파일을 찾을 수 없습니다.');
end
fprintf('총 %d개의 CSV 파일을 발견했습니다. 순차 처리를 시작합니다.\n', n_files);
% =========================================================
%  2) 각 파일별 루프 실행
% =========================================================
for i_file = 1:n_files
    current_folder   = file_list(i_file).folder;
    current_csv_name = string(file_list(i_file).name);
    full_csv_path    = fullfile(current_folder, current_csv_name);
    base_name = erase(current_csv_name, ".csv");
    struct_var_name = matlab.lang.makeValidName(base_name); 
    mat_filename_struct = base_name + ".mat";
    full_save_path_struct = fullfile(current_folder, mat_filename_struct);
    
    fprintf('\n--------------------------------------------------------------\n');
    fprintf('[%d/%d] 처리 중: %s\n', i_file, n_files, full_csv_path);
    
    try
        T = readtable(full_csv_path);
    catch
        warning('파일 읽기 실패: %s', full_csv_path); continue;
    end
    
    need_vars = ["subject","testSession","day","modeling_used", ...
                 "stride_idx","t","y_true_Fx","y_pred_Fx","y_true_Fz","y_pred_Fz"];
    if ~all(ismember(need_vars, string(T.Properties.VariableNames)))
        warning('필수 컬럼 누락으로 건너뜀: %s', current_csv_name); continue; 
    end
    
    T.subject = string(T.subject);
    T.session = string(T.testSession);
    T.day     = string(T.day);
    mod_str   = string(T.modeling_used);
    is_true   = (mod_str == "True") | (mod_str == "true") | (mod_str == "1");
    T.modeling_flag = is_true;
    
    subjects = unique(T.subject).';
    plot_data = struct(); 
    
    for s_idx = 1:numel(subjects)
        subject_id = subjects(s_idx);
        subj_field = matlab.lang.makeValidName(char(subject_id));
        
        mask_subj = (T.subject == subject_id);
        T_subj    = T(mask_subj, :);
        comb_table = unique(T_subj(:, {'session','day','modeling_flag'}), 'rows');
        n_combos   = height(comb_table);
        
        plot_data.(subj_field) = struct( ...
            'subject', cell(1,n_combos), 'session', cell(1,n_combos), ...
            'day', cell(1,n_combos), 'modeling_used', cell(1,n_combos), ...
            'stride_list', cell(1,n_combos), 't', cell(1,n_combos), ...
            'Fx_true', cell(1,n_combos), 'Fx_pred', cell(1,n_combos), ...
            'Fz_true', cell(1,n_combos), 'Fz_pred', cell(1,n_combos), ...
            'Fx_true_mean', cell(1,n_combos), 'Fx_pred_mean', cell(1,n_combos), ...
            'Fz_true_mean', cell(1,n_combos), 'Fz_pred_mean', cell(1,n_combos), ...
            'metrics', cell(1,n_combos) );
        
        for c_idx = 1:n_combos
            session_id = comb_table.session(c_idx);
            day_id     = comb_table.day(c_idx);
            mod_case   = comb_table.modeling_flag(c_idx);
            
            mask_case = (T_subj.session == session_id) & (T_subj.day == day_id) & (T_subj.modeling_flag == mod_case);
            T_sel = T_subj(mask_case, :);
            stride_all = unique(T_sel.stride_idx).';
            if isempty(stride_idx_list), stride_list = stride_all;
            else, stride_list = intersect(unique(stride_idx_list), stride_all); end
            stride_list = sort(stride_list);
            
            n_stride = numel(stride_list);
            if n_stride == 0, continue; end
            
            t_all = cell(n_stride, 1);
            Fx_true_all = cell(n_stride,1); Fx_pred_all = cell(n_stride,1);
            Fz_true_all = cell(n_stride,1); Fz_pred_all = cell(n_stride,1);
            
            for k=1:n_stride
                sid = stride_list(k);
                T_k = T_sel(T_sel.stride_idx == sid, :);
                [t_sorted, idx_sort] = sort(T_k.t);
                t_all{k} = t_sorted;
                Fx_true_all{k} = T_k.y_true_Fx(idx_sort); Fx_pred_all{k} = T_k.y_pred_Fx(idx_sort);
                Fz_true_all{k} = T_k.y_true_Fz(idx_sort); Fz_pred_all{k} = T_k.y_pred_Fz(idx_sort);
            end
            
            N = numel(t_all{1});
            Fx_true_mat = zeros(N, n_stride); Fx_pred_mat = zeros(N, n_stride);
            Fz_true_mat = zeros(N, n_stride); Fz_pred_mat = zeros(N, n_stride);
            
            skip_combo = false;
            for k=1:n_stride
                if numel(t_all{k}) ~= N, skip_combo=true; break; end
                Fx_true_mat(:,k) = Fx_true_all{k}; Fx_pred_mat(:,k) = Fx_pred_all{k};
                Fz_true_mat(:,k) = Fz_true_all{k}; Fz_pred_mat(:,k) = Fz_pred_all{k};
            end
            if skip_combo, continue; end
            
            t_mean = t_all{1};
            Fx_true_mean = mean(Fx_true_mat,2); Fx_pred_mean = mean(Fx_pred_mat,2);
            Fz_true_mean = mean(Fz_true_mat,2); Fz_pred_mean = mean(Fz_pred_mat,2);
            
            % --- Metrics Calculation ---
            % [기본]
            Fx_RMSE = zeros(1,n_stride); Fz_RMSE = zeros(1,n_stride);
            Fx_NRMSE_range = NaN(1,n_stride); Fz_NRMSE_range = NaN(1,n_stride);
            Fx_Corr = NaN(1, n_stride); Fz_Corr = NaN(1, n_stride); All_Corr = NaN(1, n_stride);
            
            % [Phase 1] Flex / Ext
            Fx_NRMSE_Flex = NaN(1,n_stride); Fx_NRMSE_Ext = NaN(1,n_stride);
            Fz_NRMSE_Flex = NaN(1,n_stride); Fz_NRMSE_Ext = NaN(1,n_stride);
            Fx_Corr_Flex  = NaN(1,n_stride); Fx_Corr_Ext  = NaN(1,n_stride);
            Fz_Corr_Flex  = NaN(1,n_stride); Fz_Corr_Ext  = NaN(1,n_stride);
            
            % [Phase 2] Ipsi / Contra
            Fx_NRMSE_Ipsi = NaN(1,n_stride); Fx_NRMSE_Contra = NaN(1,n_stride);
            Fz_NRMSE_Ipsi = NaN(1,n_stride); Fz_NRMSE_Contra = NaN(1,n_stride);
            Fx_Corr_Ipsi  = NaN(1,n_stride); Fx_Corr_Contra  = NaN(1,n_stride);
            Fz_Corr_Ipsi  = NaN(1,n_stride); Fz_Corr_Contra  = NaN(1,n_stride);
            
            % ===== [NEW] Peak & Integral per stride =====
            Fx_Pos_Peak_True_arr = NaN(1,n_stride); Fx_Pos_Peak_Pred_arr = NaN(1,n_stride);
            Fx_Neg_Peak_True_arr = NaN(1,n_stride); Fx_Neg_Peak_Pred_arr = NaN(1,n_stride);
            Fz_Pos_Peak_True_arr = NaN(1,n_stride); Fz_Pos_Peak_Pred_arr = NaN(1,n_stride);
            Fz_Neg_Peak_True_arr = NaN(1,n_stride); Fz_Neg_Peak_Pred_arr = NaN(1,n_stride);
            
            Fx_Pos_Integral_True_arr = NaN(1,n_stride); Fx_Pos_Integral_Pred_arr = NaN(1,n_stride);
            Fx_Neg_Integral_True_arr = NaN(1,n_stride); Fx_Neg_Integral_Pred_arr = NaN(1,n_stride);
            Fz_Pos_Integral_True_arr = NaN(1,n_stride); Fz_Pos_Integral_Pred_arr = NaN(1,n_stride);
            Fz_Neg_Integral_True_arr = NaN(1,n_stride); Fz_Neg_Integral_Pred_arr = NaN(1,n_stride);
            
            Fx_Pos_Peak_Err_arr = NaN(1,n_stride); Fx_Neg_Peak_Err_arr = NaN(1,n_stride);
            Fz_Pos_Peak_Err_arr = NaN(1,n_stride); Fz_Neg_Peak_Err_arr = NaN(1,n_stride);
            Fx_Pos_Integral_Err_arr = NaN(1,n_stride); Fx_Neg_Integral_Err_arr = NaN(1,n_stride);
            Fz_Pos_Integral_Err_arr = NaN(1,n_stride); Fz_Neg_Integral_Err_arr = NaN(1,n_stride);
            % ===== [END NEW INIT] =====
            
            for k=1:n_stride
                sid = stride_list(k);
                
                % (A) Full Cycle
                R_mat_fx = corrcoef(Fx_true_mat(:,k), Fx_pred_mat(:,k));
                if numel(R_mat_fx)>1, Fx_Corr(k) = R_mat_fx(1,2); end
                R_mat_fz = corrcoef(Fz_true_mat(:,k), Fz_pred_mat(:,k));
                if numel(R_mat_fz)>1, Fz_Corr(k) = R_mat_fz(1,2); end
                if ~isnan(Fx_Corr(k)) && ~isnan(Fz_Corr(k)), All_Corr(k) = (Fx_Corr(k)+Fz_Corr(k))/2; end
                
                err_fx = Fx_pred_mat(:,k) - Fx_true_mat(:,k);
                Fx_RMSE(k) = sqrt(mean(err_fx.^2));
                r_fx = max(Fx_true_mat(:,k)) - min(Fx_true_mat(:,k));
                if r_fx>0, Fx_NRMSE_range(k) = Fx_RMSE(k)/r_fx; end
                
                err_fz = Fz_pred_mat(:,k) - Fz_true_mat(:,k);
                Fz_RMSE(k) = sqrt(mean(err_fz.^2));
                r_fz = max(Fz_true_mat(:,k)) - min(Fz_true_mat(:,k));
                if r_fz>0, Fz_NRMSE_range(k) = Fz_RMSE(k)/r_fz; end
                
                if N == 101
                    % (B) Flexion / Extension
                    t_x_flx = Fx_true_mat(idx_flex, k); p_x_flx = Fx_pred_mat(idx_flex, k);
                    t_z_flx = Fz_true_mat(idx_flex, k); p_z_flx = Fz_pred_mat(idx_flex, k);
                    if r_fx>0, Fx_NRMSE_Flex(k) = sqrt(mean((t_x_flx - p_x_flx).^2))/r_fx; end
                    if r_fz>0, Fz_NRMSE_Flex(k) = sqrt(mean((t_z_flx - p_z_flx).^2))/r_fz; end
                    r = corrcoef(t_x_flx, p_x_flx); if numel(r)>1, Fx_Corr_Flex(k)=r(1,2); end
                    r = corrcoef(t_z_flx, p_z_flx); if numel(r)>1, Fz_Corr_Flex(k)=r(1,2); end
                    
                    t_x_ext = Fx_true_mat(idx_ext, k); p_x_ext = Fx_pred_mat(idx_ext, k);
                    t_z_ext = Fz_true_mat(idx_ext, k); p_z_ext = Fz_pred_mat(idx_ext, k);
                    if r_fx>0, Fx_NRMSE_Ext(k) = sqrt(mean((t_x_ext - p_x_ext).^2))/r_fx; end
                    if r_fz>0, Fz_NRMSE_Ext(k) = sqrt(mean((t_z_ext - p_z_ext).^2))/r_fz; end
                    r = corrcoef(t_x_ext, p_x_ext); if numel(r)>1, Fx_Corr_Ext(k)=r(1,2); end
                    r = corrcoef(t_z_ext, p_z_ext); if numel(r)>1, Fz_Corr_Ext(k)=r(1,2); end
                    
                    % (C) Ipsilateral / Contralateral
                    t_x_ipsi = Fx_true_mat(idx_ipsi, k); p_x_ipsi = Fx_pred_mat(idx_ipsi, k);
                    t_z_ipsi = Fz_true_mat(idx_ipsi, k); p_z_ipsi = Fz_pred_mat(idx_ipsi, k);
                    if r_fx>0, Fx_NRMSE_Ipsi(k) = sqrt(mean((t_x_ipsi - p_x_ipsi).^2))/r_fx; end
                    if r_fz>0, Fz_NRMSE_Ipsi(k) = sqrt(mean((t_z_ipsi - p_z_ipsi).^2))/r_fz; end
                    r = corrcoef(t_x_ipsi, p_x_ipsi); if numel(r)>1, Fx_Corr_Ipsi(k)=r(1,2); end
                    r = corrcoef(t_z_ipsi, p_z_ipsi); if numel(r)>1, Fz_Corr_Ipsi(k)=r(1,2); end
                    
                    t_x_contra = Fx_true_mat(idx_contra, k); p_x_contra = Fx_pred_mat(idx_contra, k);
                    t_z_contra = Fz_true_mat(idx_contra, k); p_z_contra = Fz_pred_mat(idx_contra, k);
                    if r_fx>0, Fx_NRMSE_Contra(k) = sqrt(mean((t_x_contra - p_x_contra).^2))/r_fx; end
                    if r_fz>0, Fz_NRMSE_Contra(k) = sqrt(mean((t_z_contra - p_z_contra).^2))/r_fz; end
                    r = corrcoef(t_x_contra, p_x_contra); if numel(r)>1, Fx_Corr_Contra(k)=r(1,2); end
                    r = corrcoef(t_z_contra, p_z_contra); if numel(r)>1, Fz_Corr_Contra(k)=r(1,2); end
                end
                
                % ===== [NEW] Peak & Integral Calculation =====
                % --- Get actual time vector from stride_kinematics_arm ---
                try
                    time_vec = stride_kinematics_arm.L_Wrist.(char(subject_id)).(char(session_id)).(char(day_id)).time{sid, 1};
                catch
                    warning('stride_kinematics_arm 시간 데이터 접근 실패: %s / %s / %s / stride %d', ...
                        subject_id, session_id, day_id, sid);
                    continue;  % 이 stride는 NaN으로 남김
                end
                
                % --- Peaks ---
                fx_t = Fx_true_mat(:,k);  fx_p = Fx_pred_mat(:,k);
                fz_t = Fz_true_mat(:,k);  fz_p = Fz_pred_mat(:,k);
                
                Fx_Pos_Peak_True_arr(k) = max(fx_t);
                Fx_Pos_Peak_Pred_arr(k) = max(fx_p);
                Fx_Neg_Peak_True_arr(k) = min(fx_t);
                Fx_Neg_Peak_Pred_arr(k) = min(fx_p);
                
                Fz_Pos_Peak_True_arr(k) = max(fz_t);
                Fz_Pos_Peak_Pred_arr(k) = max(fz_p);
                Fz_Neg_Peak_True_arr(k) = min(fz_t);
                Fz_Neg_Peak_Pred_arr(k) = min(fz_p);
                
                % --- Peak Errors (absolute) ---
                Fx_Pos_Peak_Err_arr(k) = abs(Fx_Pos_Peak_True_arr(k) - Fx_Pos_Peak_Pred_arr(k));
                Fx_Neg_Peak_Err_arr(k) = abs(Fx_Neg_Peak_True_arr(k) - Fx_Neg_Peak_Pred_arr(k));
                Fz_Pos_Peak_Err_arr(k) = abs(Fz_Pos_Peak_True_arr(k) - Fz_Pos_Peak_Pred_arr(k));
                Fz_Neg_Peak_Err_arr(k) = abs(Fz_Neg_Peak_True_arr(k) - Fz_Neg_Peak_Pred_arr(k));
                
                % --- Integrals (trapz with actual time) ---
                % Fx True
                sig = fx_t;
                sig_pos = sig; sig_pos(sig <= 0) = 0;
                sig_neg = sig; sig_neg(sig >= 0) = 0;
                Fx_Pos_Integral_True_arr(k) = trapz(time_vec, sig_pos);
                Fx_Neg_Integral_True_arr(k) = trapz(time_vec, sig_neg);  % 음수 그대로
                
                % Fx Pred
                sig = fx_p;
                sig_pos = sig; sig_pos(sig <= 0) = 0;
                sig_neg = sig; sig_neg(sig >= 0) = 0;
                Fx_Pos_Integral_Pred_arr(k) = trapz(time_vec, sig_pos);
                Fx_Neg_Integral_Pred_arr(k) = trapz(time_vec, sig_neg);
                
                % Fz True
                sig = fz_t;
                sig_pos = sig; sig_pos(sig <= 0) = 0;
                sig_neg = sig; sig_neg(sig >= 0) = 0;
                Fz_Pos_Integral_True_arr(k) = trapz(time_vec, sig_pos);
                Fz_Neg_Integral_True_arr(k) = trapz(time_vec, sig_neg);
                
                % Fz Pred
                sig = fz_p;
                sig_pos = sig; sig_pos(sig <= 0) = 0;
                sig_neg = sig; sig_neg(sig >= 0) = 0;
                Fz_Pos_Integral_Pred_arr(k) = trapz(time_vec, sig_pos);
                Fz_Neg_Integral_Pred_arr(k) = trapz(time_vec, sig_neg);
                
                % --- Integral Errors (absolute) ---
                Fx_Pos_Integral_Err_arr(k) = abs(Fx_Pos_Integral_True_arr(k) - Fx_Pos_Integral_Pred_arr(k));
                Fx_Neg_Integral_Err_arr(k) = abs(Fx_Neg_Integral_True_arr(k) - Fx_Neg_Integral_Pred_arr(k));
                Fz_Pos_Integral_Err_arr(k) = abs(Fz_Pos_Integral_True_arr(k) - Fz_Pos_Integral_Pred_arr(k));
                Fz_Neg_Integral_Err_arr(k) = abs(Fz_Neg_Integral_True_arr(k) - Fz_Neg_Integral_Pred_arr(k));
                % ===== [END NEW] =====
            end
            
            % Save Metrics Structure
            met = struct();
            met.Fx_Corr_each_stride = Fx_Corr; met.Fz_Corr_each_stride = Fz_Corr; met.All_Corr_each_stride = All_Corr;
            met.Fx_RMSE_each_stride = Fx_RMSE; met.Fz_RMSE_each_stride = Fz_RMSE;
            met.Fx_NRMSE_range_each_stride = Fx_NRMSE_range; met.Fz_NRMSE_range_each_stride = Fz_NRMSE_range;
            
            % Phase 1: Flex/Ext
            met.Fx_NRMSE_Flex_each = Fx_NRMSE_Flex; met.Fx_NRMSE_Ext_each = Fx_NRMSE_Ext;
            met.Fz_NRMSE_Flex_each = Fz_NRMSE_Flex; met.Fz_NRMSE_Ext_each = Fz_NRMSE_Ext;
            met.Fx_Corr_Flex_each  = Fx_Corr_Flex;  met.Fx_Corr_Ext_each  = Fx_Corr_Ext;
            met.Fz_Corr_Flex_each  = Fz_Corr_Flex;  met.Fz_Corr_Ext_each  = Fz_Corr_Ext;
            
            % Phase 2: Ipsi/Contra
            met.Fx_NRMSE_Ipsi_each = Fx_NRMSE_Ipsi; met.Fx_NRMSE_Contra_each = Fx_NRMSE_Contra;
            met.Fz_NRMSE_Ipsi_each = Fz_NRMSE_Ipsi; met.Fz_NRMSE_Contra_each = Fz_NRMSE_Contra;
            met.Fx_Corr_Ipsi_each  = Fx_Corr_Ipsi;  met.Fx_Corr_Contra_each  = Fx_Corr_Contra;
            met.Fz_Corr_Ipsi_each  = Fz_Corr_Ipsi;  met.Fz_Corr_Contra_each  = Fz_Corr_Contra;
            
            % ===== [NEW] Peak & Integral into met =====
            met.Fx_Pos_Peak_True_each = Fx_Pos_Peak_True_arr;
            met.Fx_Pos_Peak_Pred_each = Fx_Pos_Peak_Pred_arr;
            met.Fx_Neg_Peak_True_each = Fx_Neg_Peak_True_arr;
            met.Fx_Neg_Peak_Pred_each = Fx_Neg_Peak_Pred_arr;
            met.Fz_Pos_Peak_True_each = Fz_Pos_Peak_True_arr;
            met.Fz_Pos_Peak_Pred_each = Fz_Pos_Peak_Pred_arr;
            met.Fz_Neg_Peak_True_each = Fz_Neg_Peak_True_arr;
            met.Fz_Neg_Peak_Pred_each = Fz_Neg_Peak_Pred_arr;
            
            met.Fx_Pos_Integral_True_each = Fx_Pos_Integral_True_arr;
            met.Fx_Pos_Integral_Pred_each = Fx_Pos_Integral_Pred_arr;
            met.Fx_Neg_Integral_True_each = Fx_Neg_Integral_True_arr;
            met.Fx_Neg_Integral_Pred_each = Fx_Neg_Integral_Pred_arr;
            met.Fz_Pos_Integral_True_each = Fz_Pos_Integral_True_arr;
            met.Fz_Pos_Integral_Pred_each = Fz_Pos_Integral_Pred_arr;
            met.Fz_Neg_Integral_True_each = Fz_Neg_Integral_True_arr;
            met.Fz_Neg_Integral_Pred_each = Fz_Neg_Integral_Pred_arr;
            
            met.Fx_Pos_Peak_Err_each = Fx_Pos_Peak_Err_arr;
            met.Fx_Neg_Peak_Err_each = Fx_Neg_Peak_Err_arr;
            met.Fz_Pos_Peak_Err_each = Fz_Pos_Peak_Err_arr;
            met.Fz_Neg_Peak_Err_each = Fz_Neg_Peak_Err_arr;
            met.Fx_Pos_Integral_Err_each = Fx_Pos_Integral_Err_arr;
            met.Fx_Neg_Integral_Err_each = Fx_Neg_Integral_Err_arr;
            met.Fz_Pos_Integral_Err_each = Fz_Pos_Integral_Err_arr;
            met.Fz_Neg_Integral_Err_each = Fz_Neg_Integral_Err_arr;
            % ===== [END NEW] =====
            
            plot_data.(subj_field)(c_idx).metrics = met;
            plot_data.(subj_field)(c_idx).t = t_mean;
            plot_data.(subj_field)(c_idx).Fx_true = Fx_true_mat; plot_data.(subj_field)(c_idx).Fx_pred = Fx_pred_mat;
            plot_data.(subj_field)(c_idx).Fz_true = Fz_true_mat; plot_data.(subj_field)(c_idx).Fz_pred = Fz_pred_mat;
        
        
            plot_data.(subj_field)(c_idx).subject = subject_id;
            plot_data.(subj_field)(c_idx).session = session_id;
            plot_data.(subj_field)(c_idx).day = day_id;
            plot_data.(subj_field)(c_idx).modeling_used = mod_case;
            plot_data.(subj_field)(c_idx).stride_list = stride_list;

        
        
        end
    end
    
    % -------------------------------------------------
    % 2-4. [Table Generation] Summary Tables
    % -------------------------------------------------
    plot_data_loso = plot_data; 
    subj_fields = fieldnames(plot_data_loso);
    n_subj_t = numel(subj_fields);
    
    % (A) Initialize - Per-subject  [기존]
    mean_nrmse_all_subj=NaN(n_subj_t,1); std_nrmse_all_subj=NaN(n_subj_t,1);
    mean_nrmse_fx_subj=NaN(n_subj_t,1); std_nrmse_fx_subj=NaN(n_subj_t,1);
    mean_nrmse_fz_subj=NaN(n_subj_t,1); std_nrmse_fz_subj=NaN(n_subj_t,1);
    mean_corr_all_subj=NaN(n_subj_t,1); std_corr_all_subj=NaN(n_subj_t,1);
    mean_corr_fx_subj=NaN(n_subj_t,1);  std_corr_fx_subj=NaN(n_subj_t,1);
    mean_corr_fz_subj=NaN(n_subj_t,1);  std_corr_fz_subj=NaN(n_subj_t,1);
    
    % Flex/Ext
    mean_nrmse_fx_flx_subj=NaN(n_subj_t,1); std_nrmse_fx_flx_subj=NaN(n_subj_t,1);
    mean_nrmse_fz_flx_subj=NaN(n_subj_t,1); std_nrmse_fz_flx_subj=NaN(n_subj_t,1);
    mean_nrmse_fx_ext_subj=NaN(n_subj_t,1); std_nrmse_fx_ext_subj=NaN(n_subj_t,1);
    mean_nrmse_fz_ext_subj=NaN(n_subj_t,1); std_nrmse_fz_ext_subj=NaN(n_subj_t,1);
    mean_corr_fx_flx_subj=NaN(n_subj_t,1); std_corr_fx_flx_subj=NaN(n_subj_t,1);
    mean_corr_fz_flx_subj=NaN(n_subj_t,1); std_corr_fz_flx_subj=NaN(n_subj_t,1);
    mean_corr_fx_ext_subj=NaN(n_subj_t,1); std_corr_fx_ext_subj=NaN(n_subj_t,1);
    mean_corr_fz_ext_subj=NaN(n_subj_t,1); std_corr_fz_ext_subj=NaN(n_subj_t,1);
    
    % Ipsi/Contra
    mean_nrmse_fx_ipsi_subj=NaN(n_subj_t,1); std_nrmse_fx_ipsi_subj=NaN(n_subj_t,1);
    mean_nrmse_fz_ipsi_subj=NaN(n_subj_t,1); std_nrmse_fz_ipsi_subj=NaN(n_subj_t,1);
    mean_nrmse_fx_contra_subj=NaN(n_subj_t,1); std_nrmse_fx_contra_subj=NaN(n_subj_t,1);
    mean_nrmse_fz_contra_subj=NaN(n_subj_t,1); std_nrmse_fz_contra_subj=NaN(n_subj_t,1);
    mean_corr_fx_ipsi_subj=NaN(n_subj_t,1); std_corr_fx_ipsi_subj=NaN(n_subj_t,1);
    mean_corr_fz_ipsi_subj=NaN(n_subj_t,1); std_corr_fz_ipsi_subj=NaN(n_subj_t,1);
    mean_corr_fx_contra_subj=NaN(n_subj_t,1); std_corr_fx_contra_subj=NaN(n_subj_t,1);
    mean_corr_fz_contra_subj=NaN(n_subj_t,1); std_corr_fz_contra_subj=NaN(n_subj_t,1);
    
    % ===== [NEW] Peak & Integral per-subject arrays =====
    % --- trueTable용: True/Pred 값의 mean/std ---
    mean_fx_pp_true_subj=NaN(n_subj_t,1); std_fx_pp_true_subj=NaN(n_subj_t,1);
    mean_fx_pp_pred_subj=NaN(n_subj_t,1); std_fx_pp_pred_subj=NaN(n_subj_t,1);
    mean_fx_np_true_subj=NaN(n_subj_t,1); std_fx_np_true_subj=NaN(n_subj_t,1);
    mean_fx_np_pred_subj=NaN(n_subj_t,1); std_fx_np_pred_subj=NaN(n_subj_t,1);
    mean_fz_pp_true_subj=NaN(n_subj_t,1); std_fz_pp_true_subj=NaN(n_subj_t,1);
    mean_fz_pp_pred_subj=NaN(n_subj_t,1); std_fz_pp_pred_subj=NaN(n_subj_t,1);
    mean_fz_np_true_subj=NaN(n_subj_t,1); std_fz_np_true_subj=NaN(n_subj_t,1);
    mean_fz_np_pred_subj=NaN(n_subj_t,1); std_fz_np_pred_subj=NaN(n_subj_t,1);
    
    mean_fx_pi_true_subj=NaN(n_subj_t,1); std_fx_pi_true_subj=NaN(n_subj_t,1);
    mean_fx_pi_pred_subj=NaN(n_subj_t,1); std_fx_pi_pred_subj=NaN(n_subj_t,1);
    mean_fx_ni_true_subj=NaN(n_subj_t,1); std_fx_ni_true_subj=NaN(n_subj_t,1);
    mean_fx_ni_pred_subj=NaN(n_subj_t,1); std_fx_ni_pred_subj=NaN(n_subj_t,1);
    mean_fz_pi_true_subj=NaN(n_subj_t,1); std_fz_pi_true_subj=NaN(n_subj_t,1);
    mean_fz_pi_pred_subj=NaN(n_subj_t,1); std_fz_pi_pred_subj=NaN(n_subj_t,1);
    mean_fz_ni_true_subj=NaN(n_subj_t,1); std_fz_ni_true_subj=NaN(n_subj_t,1);
    mean_fz_ni_pred_subj=NaN(n_subj_t,1); std_fz_ni_pred_subj=NaN(n_subj_t,1);
    
    % --- errTable용: Error의 mean/std ---
    mean_fx_pp_err_subj=NaN(n_subj_t,1); std_fx_pp_err_subj=NaN(n_subj_t,1);
    mean_fx_np_err_subj=NaN(n_subj_t,1); std_fx_np_err_subj=NaN(n_subj_t,1);
    mean_fz_pp_err_subj=NaN(n_subj_t,1); std_fz_pp_err_subj=NaN(n_subj_t,1);
    mean_fz_np_err_subj=NaN(n_subj_t,1); std_fz_np_err_subj=NaN(n_subj_t,1);
    mean_fx_pi_err_subj=NaN(n_subj_t,1); std_fx_pi_err_subj=NaN(n_subj_t,1);
    mean_fx_ni_err_subj=NaN(n_subj_t,1); std_fx_ni_err_subj=NaN(n_subj_t,1);
    mean_fz_pi_err_subj=NaN(n_subj_t,1); std_fz_pi_err_subj=NaN(n_subj_t,1);
    mean_fz_ni_err_subj=NaN(n_subj_t,1); std_fz_ni_err_subj=NaN(n_subj_t,1);
    % ===== [END NEW INIT] =====
    
    % (B) Initialize - Accumulators (Total)
    all_nrmse_fx=[]; all_nrmse_fz=[]; all_corr_all=[]; all_corr_fx=[]; all_corr_fz=[];
    all_nrmse_fx_flx=[]; all_nrmse_fz_flx=[]; all_nrmse_fx_ext=[]; all_nrmse_fz_ext=[];
    all_corr_fx_flx=[];  all_corr_fz_flx=[];  all_corr_fx_ext=[];  all_corr_fz_ext=[];
    all_nrmse_fx_ipsi=[]; all_nrmse_fz_ipsi=[]; all_nrmse_fx_contra=[]; all_nrmse_fz_contra=[];
    all_corr_fx_ipsi=[];  all_corr_fz_ipsi=[];  all_corr_fx_contra=[];  all_corr_fz_contra=[];
    
    % (C) Loop
    for s_idx = 1:n_subj_t
        subj_field = subj_fields{s_idx};
        cases = plot_data_loso.(subj_field);
        
        nrmse_fx_subj_vals=[]; nrmse_fz_subj_vals=[];
        corr_all_subj_vals=[]; corr_fx_subj_vals=[]; corr_fz_subj_vals=[];
        
        nrmse_fx_flx_v=[]; nrmse_fz_flx_v=[]; nrmse_fx_ext_v=[]; nrmse_fz_ext_v=[];
        corr_fx_flx_v=[];  corr_fz_flx_v=[];  corr_fx_ext_v=[];  corr_fz_ext_v=[];
        
        nrmse_fx_ipsi_v=[]; nrmse_fz_ipsi_v=[]; nrmse_fx_contra_v=[]; nrmse_fz_contra_v=[];
        corr_fx_ipsi_v=[];  corr_fz_ipsi_v=[];  corr_fx_contra_v=[];  corr_fz_contra_v=[];
        
        % ===== [NEW] Accumulators for Peak & Integral =====
        fx_pp_true_v=[]; fx_pp_pred_v=[]; fx_np_true_v=[]; fx_np_pred_v=[];
        fz_pp_true_v=[]; fz_pp_pred_v=[]; fz_np_true_v=[]; fz_np_pred_v=[];
        fx_pi_true_v=[]; fx_pi_pred_v=[]; fx_ni_true_v=[]; fx_ni_pred_v=[];
        fz_pi_true_v=[]; fz_pi_pred_v=[]; fz_ni_true_v=[]; fz_ni_pred_v=[];
        fx_pp_err_v=[]; fx_np_err_v=[]; fz_pp_err_v=[]; fz_np_err_v=[];
        fx_pi_err_v=[]; fx_ni_err_v=[]; fz_pi_err_v=[]; fz_ni_err_v=[];
        % ===== [END NEW] =====
        
        for c_idx = 1:numel(cases)
            met = cases(c_idx).metrics;
            if isempty(met), continue; end
            
            % Basic
            nrmse_fx_subj_vals = [nrmse_fx_subj_vals, met.Fx_NRMSE_range_each_stride(:).'];
            nrmse_fz_subj_vals = [nrmse_fz_subj_vals, met.Fz_NRMSE_range_each_stride(:).'];
            corr_all_subj_vals = [corr_all_subj_vals, met.All_Corr_each_stride(:).'];
            corr_fx_subj_vals = [corr_fx_subj_vals, met.Fx_Corr_each_stride(:).'];
            corr_fz_subj_vals = [corr_fz_subj_vals, met.Fz_Corr_each_stride(:).'];
            
            % Flex/Ext
            nrmse_fx_flx_v = [nrmse_fx_flx_v, met.Fx_NRMSE_Flex_each(:).'];
            nrmse_fz_flx_v = [nrmse_fz_flx_v, met.Fz_NRMSE_Flex_each(:).'];
            nrmse_fx_ext_v = [nrmse_fx_ext_v, met.Fx_NRMSE_Ext_each(:).'];
            nrmse_fz_ext_v = [nrmse_fz_ext_v, met.Fz_NRMSE_Ext_each(:).'];
            corr_fx_flx_v = [corr_fx_flx_v, met.Fx_Corr_Flex_each(:).'];
            corr_fz_flx_v = [corr_fz_flx_v, met.Fz_Corr_Flex_each(:).'];
            corr_fx_ext_v = [corr_fx_ext_v, met.Fx_Corr_Ext_each(:).'];
            corr_fz_ext_v = [corr_fz_ext_v, met.Fz_Corr_Ext_each(:).'];
            
            % Ipsi/Contra
            nrmse_fx_ipsi_v = [nrmse_fx_ipsi_v, met.Fx_NRMSE_Ipsi_each(:).'];
            nrmse_fz_ipsi_v = [nrmse_fz_ipsi_v, met.Fz_NRMSE_Ipsi_each(:).'];
            nrmse_fx_contra_v = [nrmse_fx_contra_v, met.Fx_NRMSE_Contra_each(:).'];
            nrmse_fz_contra_v = [nrmse_fz_contra_v, met.Fz_NRMSE_Contra_each(:).'];
            corr_fx_ipsi_v = [corr_fx_ipsi_v, met.Fx_Corr_Ipsi_each(:).'];
            corr_fz_ipsi_v = [corr_fz_ipsi_v, met.Fz_Corr_Ipsi_each(:).'];
            corr_fx_contra_v = [corr_fx_contra_v, met.Fx_Corr_Contra_each(:).'];
            corr_fz_contra_v = [corr_fz_contra_v, met.Fz_Corr_Contra_each(:).'];
            
            % ===== [NEW] Accumulate Peak & Integral =====
            if isfield(met, 'Fx_Pos_Peak_True_each')
                fx_pp_true_v = [fx_pp_true_v, met.Fx_Pos_Peak_True_each(:).'];
                fx_pp_pred_v = [fx_pp_pred_v, met.Fx_Pos_Peak_Pred_each(:).'];
                fx_np_true_v = [fx_np_true_v, met.Fx_Neg_Peak_True_each(:).'];
                fx_np_pred_v = [fx_np_pred_v, met.Fx_Neg_Peak_Pred_each(:).'];
                fz_pp_true_v = [fz_pp_true_v, met.Fz_Pos_Peak_True_each(:).'];
                fz_pp_pred_v = [fz_pp_pred_v, met.Fz_Pos_Peak_Pred_each(:).'];
                fz_np_true_v = [fz_np_true_v, met.Fz_Neg_Peak_True_each(:).'];
                fz_np_pred_v = [fz_np_pred_v, met.Fz_Neg_Peak_Pred_each(:).'];
                
                fx_pi_true_v = [fx_pi_true_v, met.Fx_Pos_Integral_True_each(:).'];
                fx_pi_pred_v = [fx_pi_pred_v, met.Fx_Pos_Integral_Pred_each(:).'];
                fx_ni_true_v = [fx_ni_true_v, met.Fx_Neg_Integral_True_each(:).'];
                fx_ni_pred_v = [fx_ni_pred_v, met.Fx_Neg_Integral_Pred_each(:).'];
                fz_pi_true_v = [fz_pi_true_v, met.Fz_Pos_Integral_True_each(:).'];
                fz_pi_pred_v = [fz_pi_pred_v, met.Fz_Pos_Integral_Pred_each(:).'];
                fz_ni_true_v = [fz_ni_true_v, met.Fz_Neg_Integral_True_each(:).'];
                fz_ni_pred_v = [fz_ni_pred_v, met.Fz_Neg_Integral_Pred_each(:).'];
                
                fx_pp_err_v = [fx_pp_err_v, met.Fx_Pos_Peak_Err_each(:).'];
                fx_np_err_v = [fx_np_err_v, met.Fx_Neg_Peak_Err_each(:).'];
                fz_pp_err_v = [fz_pp_err_v, met.Fz_Pos_Peak_Err_each(:).'];
                fz_np_err_v = [fz_np_err_v, met.Fz_Neg_Peak_Err_each(:).'];
                fx_pi_err_v = [fx_pi_err_v, met.Fx_Pos_Integral_Err_each(:).'];
                fx_ni_err_v = [fx_ni_err_v, met.Fx_Neg_Integral_Err_each(:).'];
                fz_pi_err_v = [fz_pi_err_v, met.Fz_Pos_Integral_Err_each(:).'];
                fz_ni_err_v = [fz_ni_err_v, met.Fz_Neg_Integral_Err_each(:).'];
            end
            % ===== [END NEW] =====
        end
        
        calc_subj = @(arr) [mean(arr, 'omitnan'), std(arr, 'omitnan')];
        
        % Basic
        s = calc_subj([nrmse_fx_subj_vals, nrmse_fz_subj_vals]); mean_nrmse_all_subj(s_idx)=s(1); std_nrmse_all_subj(s_idx)=s(2);
        s = calc_subj(nrmse_fx_subj_vals); mean_nrmse_fx_subj(s_idx)=s(1); std_nrmse_fx_subj(s_idx)=s(2);
        s = calc_subj(nrmse_fz_subj_vals); mean_nrmse_fz_subj(s_idx)=s(1); std_nrmse_fz_subj(s_idx)=s(2);
        s = calc_subj(corr_all_subj_vals); mean_corr_all_subj(s_idx)=s(1); std_corr_all_subj(s_idx)=s(2);
        s = calc_subj(corr_fx_subj_vals); mean_corr_fx_subj(s_idx)=s(1); std_corr_fx_subj(s_idx)=s(2);
        s = calc_subj(corr_fz_subj_vals); mean_corr_fz_subj(s_idx)=s(1); std_corr_fz_subj(s_idx)=s(2);
        
        % Flex/Ext
        s = calc_subj(nrmse_fx_flx_v); mean_nrmse_fx_flx_subj(s_idx)=s(1); std_nrmse_fx_flx_subj(s_idx)=s(2);
        s = calc_subj(nrmse_fz_flx_v); mean_nrmse_fz_flx_subj(s_idx)=s(1); std_nrmse_fz_flx_subj(s_idx)=s(2);
        s = calc_subj(nrmse_fx_ext_v); mean_nrmse_fx_ext_subj(s_idx)=s(1); std_nrmse_fx_ext_subj(s_idx)=s(2);
        s = calc_subj(nrmse_fz_ext_v); mean_nrmse_fz_ext_subj(s_idx)=s(1); std_nrmse_fz_ext_subj(s_idx)=s(2);
        s = calc_subj(corr_fx_flx_v); mean_corr_fx_flx_subj(s_idx)=s(1); std_corr_fx_flx_subj(s_idx)=s(2);
        s = calc_subj(corr_fz_flx_v); mean_corr_fz_flx_subj(s_idx)=s(1); std_corr_fz_flx_subj(s_idx)=s(2);
        s = calc_subj(corr_fx_ext_v); mean_corr_fx_ext_subj(s_idx)=s(1); std_corr_fx_ext_subj(s_idx)=s(2);
        s = calc_subj(corr_fz_ext_v); mean_corr_fz_ext_subj(s_idx)=s(1); std_corr_fz_ext_subj(s_idx)=s(2);
        
        % Ipsi/Contra
        s = calc_subj(nrmse_fx_ipsi_v); mean_nrmse_fx_ipsi_subj(s_idx)=s(1); std_nrmse_fx_ipsi_subj(s_idx)=s(2);
        s = calc_subj(nrmse_fz_ipsi_v); mean_nrmse_fz_ipsi_subj(s_idx)=s(1); std_nrmse_fz_ipsi_subj(s_idx)=s(2);
        s = calc_subj(nrmse_fx_contra_v); mean_nrmse_fx_contra_subj(s_idx)=s(1); std_nrmse_fx_contra_subj(s_idx)=s(2);
        s = calc_subj(nrmse_fz_contra_v); mean_nrmse_fz_contra_subj(s_idx)=s(1); std_nrmse_fz_contra_subj(s_idx)=s(2);
        s = calc_subj(corr_fx_ipsi_v); mean_corr_fx_ipsi_subj(s_idx)=s(1); std_corr_fx_ipsi_subj(s_idx)=s(2);
        s = calc_subj(corr_fz_ipsi_v); mean_corr_fz_ipsi_subj(s_idx)=s(1); std_corr_fz_ipsi_subj(s_idx)=s(2);
        s = calc_subj(corr_fx_contra_v); mean_corr_fx_contra_subj(s_idx)=s(1); std_corr_fx_contra_subj(s_idx)=s(2);
        s = calc_subj(corr_fz_contra_v); mean_corr_fz_contra_subj(s_idx)=s(1); std_corr_fz_contra_subj(s_idx)=s(2);
        
        % ===== [NEW] Peak & Integral per-subject calc =====
        % trueTable용
        s=calc_subj(fx_pp_true_v); mean_fx_pp_true_subj(s_idx)=s(1); std_fx_pp_true_subj(s_idx)=s(2);
        s=calc_subj(fx_pp_pred_v); mean_fx_pp_pred_subj(s_idx)=s(1); std_fx_pp_pred_subj(s_idx)=s(2);
        s=calc_subj(fx_np_true_v); mean_fx_np_true_subj(s_idx)=s(1); std_fx_np_true_subj(s_idx)=s(2);
        s=calc_subj(fx_np_pred_v); mean_fx_np_pred_subj(s_idx)=s(1); std_fx_np_pred_subj(s_idx)=s(2);
        s=calc_subj(fz_pp_true_v); mean_fz_pp_true_subj(s_idx)=s(1); std_fz_pp_true_subj(s_idx)=s(2);
        s=calc_subj(fz_pp_pred_v); mean_fz_pp_pred_subj(s_idx)=s(1); std_fz_pp_pred_subj(s_idx)=s(2);
        s=calc_subj(fz_np_true_v); mean_fz_np_true_subj(s_idx)=s(1); std_fz_np_true_subj(s_idx)=s(2);
        s=calc_subj(fz_np_pred_v); mean_fz_np_pred_subj(s_idx)=s(1); std_fz_np_pred_subj(s_idx)=s(2);
        
        s=calc_subj(fx_pi_true_v); mean_fx_pi_true_subj(s_idx)=s(1); std_fx_pi_true_subj(s_idx)=s(2);
        s=calc_subj(fx_pi_pred_v); mean_fx_pi_pred_subj(s_idx)=s(1); std_fx_pi_pred_subj(s_idx)=s(2);
        s=calc_subj(fx_ni_true_v); mean_fx_ni_true_subj(s_idx)=s(1); std_fx_ni_true_subj(s_idx)=s(2);
        s=calc_subj(fx_ni_pred_v); mean_fx_ni_pred_subj(s_idx)=s(1); std_fx_ni_pred_subj(s_idx)=s(2);
        s=calc_subj(fz_pi_true_v); mean_fz_pi_true_subj(s_idx)=s(1); std_fz_pi_true_subj(s_idx)=s(2);
        s=calc_subj(fz_pi_pred_v); mean_fz_pi_pred_subj(s_idx)=s(1); std_fz_pi_pred_subj(s_idx)=s(2);
        s=calc_subj(fz_ni_true_v); mean_fz_ni_true_subj(s_idx)=s(1); std_fz_ni_true_subj(s_idx)=s(2);
        s=calc_subj(fz_ni_pred_v); mean_fz_ni_pred_subj(s_idx)=s(1); std_fz_ni_pred_subj(s_idx)=s(2);
        
        % errTable용
        s=calc_subj(fx_pp_err_v); mean_fx_pp_err_subj(s_idx)=s(1); std_fx_pp_err_subj(s_idx)=s(2);
        s=calc_subj(fx_np_err_v); mean_fx_np_err_subj(s_idx)=s(1); std_fx_np_err_subj(s_idx)=s(2);
        s=calc_subj(fz_pp_err_v); mean_fz_pp_err_subj(s_idx)=s(1); std_fz_pp_err_subj(s_idx)=s(2);
        s=calc_subj(fz_np_err_v); mean_fz_np_err_subj(s_idx)=s(1); std_fz_np_err_subj(s_idx)=s(2);
        s=calc_subj(fx_pi_err_v); mean_fx_pi_err_subj(s_idx)=s(1); std_fx_pi_err_subj(s_idx)=s(2);
        s=calc_subj(fx_ni_err_v); mean_fx_ni_err_subj(s_idx)=s(1); std_fx_ni_err_subj(s_idx)=s(2);
        s=calc_subj(fz_pi_err_v); mean_fz_pi_err_subj(s_idx)=s(1); std_fz_pi_err_subj(s_idx)=s(2);
        s=calc_subj(fz_ni_err_v); mean_fz_ni_err_subj(s_idx)=s(1); std_fz_ni_err_subj(s_idx)=s(2);
        % ===== [END NEW] =====
        
        % Accumulate (기존)
        all_nrmse_fx=[all_nrmse_fx, nrmse_fx_subj_vals]; all_nrmse_fz=[all_nrmse_fz, nrmse_fz_subj_vals];
        all_corr_all=[all_corr_all, corr_all_subj_vals]; all_corr_fx=[all_corr_fx, corr_fx_subj_vals]; all_corr_fz=[all_corr_fz, corr_fz_subj_vals];
        all_nrmse_fx_flx=[all_nrmse_fx_flx, nrmse_fx_flx_v]; all_nrmse_fz_flx=[all_nrmse_fz_flx, nrmse_fz_flx_v];
        all_nrmse_fx_ext=[all_nrmse_fx_ext, nrmse_fx_ext_v]; all_nrmse_fz_ext=[all_nrmse_fz_ext, nrmse_fz_ext_v];
        all_corr_fx_flx=[all_corr_fx_flx, corr_fx_flx_v]; all_corr_fz_flx=[all_corr_fz_flx, corr_fz_flx_v];
        all_corr_fx_ext=[all_corr_fx_ext, corr_fx_ext_v]; all_corr_fz_ext=[all_corr_fz_ext, corr_fz_ext_v];
        all_nrmse_fx_ipsi=[all_nrmse_fx_ipsi, nrmse_fx_ipsi_v]; all_nrmse_fz_ipsi=[all_nrmse_fz_ipsi, nrmse_fz_ipsi_v];
        all_nrmse_fx_contra=[all_nrmse_fx_contra, nrmse_fx_contra_v]; all_nrmse_fz_contra=[all_nrmse_fz_contra, nrmse_fz_contra_v];
        all_corr_fx_ipsi=[all_corr_fx_ipsi, corr_fx_ipsi_v]; all_corr_fz_ipsi=[all_corr_fz_ipsi, corr_fz_ipsi_v];
        all_corr_fx_contra=[all_corr_fx_contra, corr_fx_contra_v]; all_corr_fz_contra=[all_corr_fz_contra, corr_fz_contra_v];
    end
    
    % =========================================================
    % (D) Total Calc — Subject별 평균의 mean ± SD
    % =========================================================
    
    % Basic (기존)
    tot_nrmse_all_mu = mean(mean_nrmse_all_subj, 'omitnan');
    tot_nrmse_all_sd = std(mean_nrmse_all_subj, 0, 'omitnan');
    tot_nrmse_fx_mu  = mean(mean_nrmse_fx_subj, 'omitnan');
    tot_nrmse_fx_sd  = std(mean_nrmse_fx_subj, 0, 'omitnan');
    tot_nrmse_fz_mu  = mean(mean_nrmse_fz_subj, 'omitnan');
    tot_nrmse_fz_sd  = std(mean_nrmse_fz_subj, 0, 'omitnan');
    tot_corr_all_mu  = mean(mean_corr_all_subj, 'omitnan');
    tot_corr_all_sd  = std(mean_corr_all_subj, 0, 'omitnan');
    tot_corr_fx_mu   = mean(mean_corr_fx_subj, 'omitnan');
    tot_corr_fx_sd   = std(mean_corr_fx_subj, 0, 'omitnan');
    tot_corr_fz_mu   = mean(mean_corr_fz_subj, 'omitnan');
    tot_corr_fz_sd   = std(mean_corr_fz_subj, 0, 'omitnan');
    
    % Phase: Flex/Ext (기존)
    tot_nrmse_fx_flx_mu = mean(mean_nrmse_fx_flx_subj, 'omitnan');
    tot_nrmse_fx_flx_sd = std(mean_nrmse_fx_flx_subj, 0, 'omitnan');
    tot_nrmse_fz_flx_mu = mean(mean_nrmse_fz_flx_subj, 'omitnan');
    tot_nrmse_fz_flx_sd = std(mean_nrmse_fz_flx_subj, 0, 'omitnan');
    tot_nrmse_fx_ext_mu = mean(mean_nrmse_fx_ext_subj, 'omitnan');
    tot_nrmse_fx_ext_sd = std(mean_nrmse_fx_ext_subj, 0, 'omitnan');
    tot_nrmse_fz_ext_mu = mean(mean_nrmse_fz_ext_subj, 'omitnan');
    tot_nrmse_fz_ext_sd = std(mean_nrmse_fz_ext_subj, 0, 'omitnan');
    tot_corr_fx_flx_mu  = mean(mean_corr_fx_flx_subj, 'omitnan');
    tot_corr_fx_flx_sd  = std(mean_corr_fx_flx_subj, 0, 'omitnan');
    tot_corr_fz_flx_mu  = mean(mean_corr_fz_flx_subj, 'omitnan');
    tot_corr_fz_flx_sd  = std(mean_corr_fz_flx_subj, 0, 'omitnan');
    tot_corr_fx_ext_mu  = mean(mean_corr_fx_ext_subj, 'omitnan');
    tot_corr_fx_ext_sd  = std(mean_corr_fx_ext_subj, 0, 'omitnan');
    tot_corr_fz_ext_mu  = mean(mean_corr_fz_ext_subj, 'omitnan');
    tot_corr_fz_ext_sd  = std(mean_corr_fz_ext_subj, 0, 'omitnan');
    
    % Phase: Ipsi/Contra (기존)
    tot_nrmse_fx_ipsi_mu = mean(mean_nrmse_fx_ipsi_subj, 'omitnan');
    tot_nrmse_fx_ipsi_sd = std(mean_nrmse_fx_ipsi_subj, 0, 'omitnan');
    tot_nrmse_fz_ipsi_mu = mean(mean_nrmse_fz_ipsi_subj, 'omitnan');
    tot_nrmse_fz_ipsi_sd = std(mean_nrmse_fz_ipsi_subj, 0, 'omitnan');
    tot_nrmse_fx_contra_mu = mean(mean_nrmse_fx_contra_subj, 'omitnan');
    tot_nrmse_fx_contra_sd = std(mean_nrmse_fx_contra_subj, 0, 'omitnan');
    tot_nrmse_fz_contra_mu = mean(mean_nrmse_fz_contra_subj, 'omitnan');
    tot_nrmse_fz_contra_sd = std(mean_nrmse_fz_contra_subj, 0, 'omitnan');
    tot_corr_fx_ipsi_mu  = mean(mean_corr_fx_ipsi_subj, 'omitnan');
    tot_corr_fx_ipsi_sd  = std(mean_corr_fx_ipsi_subj, 0, 'omitnan');
    tot_corr_fz_ipsi_mu  = mean(mean_corr_fz_ipsi_subj, 'omitnan');
    tot_corr_fz_ipsi_sd  = std(mean_corr_fz_ipsi_subj, 0, 'omitnan');
    tot_corr_fx_contra_mu = mean(mean_corr_fx_contra_subj, 'omitnan');
    tot_corr_fx_contra_sd = std(mean_corr_fx_contra_subj, 0, 'omitnan');
    tot_corr_fz_contra_mu = mean(mean_corr_fz_contra_subj, 'omitnan');
    tot_corr_fz_contra_sd = std(mean_corr_fz_contra_subj, 0, 'omitnan');
    
    % ===== [NEW] Total for Peak & Integral =====
    % trueTable Total
    tot_fx_pp_true_mu = mean(mean_fx_pp_true_subj,'omitnan'); tot_fx_pp_true_sd = std(mean_fx_pp_true_subj,0,'omitnan');
    tot_fx_pp_pred_mu = mean(mean_fx_pp_pred_subj,'omitnan'); tot_fx_pp_pred_sd = std(mean_fx_pp_pred_subj,0,'omitnan');
    tot_fx_np_true_mu = mean(mean_fx_np_true_subj,'omitnan'); tot_fx_np_true_sd = std(mean_fx_np_true_subj,0,'omitnan');
    tot_fx_np_pred_mu = mean(mean_fx_np_pred_subj,'omitnan'); tot_fx_np_pred_sd = std(mean_fx_np_pred_subj,0,'omitnan');
    tot_fz_pp_true_mu = mean(mean_fz_pp_true_subj,'omitnan'); tot_fz_pp_true_sd = std(mean_fz_pp_true_subj,0,'omitnan');
    tot_fz_pp_pred_mu = mean(mean_fz_pp_pred_subj,'omitnan'); tot_fz_pp_pred_sd = std(mean_fz_pp_pred_subj,0,'omitnan');
    tot_fz_np_true_mu = mean(mean_fz_np_true_subj,'omitnan'); tot_fz_np_true_sd = std(mean_fz_np_true_subj,0,'omitnan');
    tot_fz_np_pred_mu = mean(mean_fz_np_pred_subj,'omitnan'); tot_fz_np_pred_sd = std(mean_fz_np_pred_subj,0,'omitnan');
    
    tot_fx_pi_true_mu = mean(mean_fx_pi_true_subj,'omitnan'); tot_fx_pi_true_sd = std(mean_fx_pi_true_subj,0,'omitnan');
    tot_fx_pi_pred_mu = mean(mean_fx_pi_pred_subj,'omitnan'); tot_fx_pi_pred_sd = std(mean_fx_pi_pred_subj,0,'omitnan');
    tot_fx_ni_true_mu = mean(mean_fx_ni_true_subj,'omitnan'); tot_fx_ni_true_sd = std(mean_fx_ni_true_subj,0,'omitnan');
    tot_fx_ni_pred_mu = mean(mean_fx_ni_pred_subj,'omitnan'); tot_fx_ni_pred_sd = std(mean_fx_ni_pred_subj,0,'omitnan');
    tot_fz_pi_true_mu = mean(mean_fz_pi_true_subj,'omitnan'); tot_fz_pi_true_sd = std(mean_fz_pi_true_subj,0,'omitnan');
    tot_fz_pi_pred_mu = mean(mean_fz_pi_pred_subj,'omitnan'); tot_fz_pi_pred_sd = std(mean_fz_pi_pred_subj,0,'omitnan');
    tot_fz_ni_true_mu = mean(mean_fz_ni_true_subj,'omitnan'); tot_fz_ni_true_sd = std(mean_fz_ni_true_subj,0,'omitnan');
    tot_fz_ni_pred_mu = mean(mean_fz_ni_pred_subj,'omitnan'); tot_fz_ni_pred_sd = std(mean_fz_ni_pred_subj,0,'omitnan');
    
    % errTable Total (Error)
    tot_fx_pp_err_mu = mean(mean_fx_pp_err_subj,'omitnan'); tot_fx_pp_err_sd = std(mean_fx_pp_err_subj,0,'omitnan');
    tot_fx_np_err_mu = mean(mean_fx_np_err_subj,'omitnan'); tot_fx_np_err_sd = std(mean_fx_np_err_subj,0,'omitnan');
    tot_fz_pp_err_mu = mean(mean_fz_pp_err_subj,'omitnan'); tot_fz_pp_err_sd = std(mean_fz_pp_err_subj,0,'omitnan');
    tot_fz_np_err_mu = mean(mean_fz_np_err_subj,'omitnan'); tot_fz_np_err_sd = std(mean_fz_np_err_subj,0,'omitnan');
    tot_fx_pi_err_mu = mean(mean_fx_pi_err_subj,'omitnan'); tot_fx_pi_err_sd = std(mean_fx_pi_err_subj,0,'omitnan');
    tot_fx_ni_err_mu = mean(mean_fx_ni_err_subj,'omitnan'); tot_fx_ni_err_sd = std(mean_fx_ni_err_subj,0,'omitnan');
    tot_fz_pi_err_mu = mean(mean_fz_pi_err_subj,'omitnan'); tot_fz_pi_err_sd = std(mean_fz_pi_err_subj,0,'omitnan');
    tot_fz_ni_err_mu = mean(mean_fz_ni_err_subj,'omitnan'); tot_fz_ni_err_sd = std(mean_fz_ni_err_subj,0,'omitnan');
    % ===== [END NEW] =====
    
    % (E) Create Table
    RowLabel = strings(n_subj_t+1, 1);
    RowLabel(1) = "Total";
    for s_idx = 1:n_subj_t, RowLabel(s_idx+1) = string(subj_fields{s_idx}); end
    
    % Basic Cols (기존)
    Mean_nRMSE_all=[tot_nrmse_all_mu; mean_nrmse_all_subj]; Std_nRMSE_all=[tot_nrmse_all_sd; std_nrmse_all_subj];
    Mean_nRMSE_Fx=[tot_nrmse_fx_mu; mean_nrmse_fx_subj]; Std_nRMSE_Fx=[tot_nrmse_fx_sd; std_nrmse_fx_subj];
    Mean_nRMSE_Fz=[tot_nrmse_fz_mu; mean_nrmse_fz_subj]; Std_nRMSE_Fz=[tot_nrmse_fz_sd; std_nrmse_fz_subj];
    Mean_Corr_all=[tot_corr_all_mu; mean_corr_all_subj]; Std_Corr_all=[tot_corr_all_sd; std_corr_all_subj];
    Mean_Corr_Fx=[tot_corr_fx_mu; mean_corr_fx_subj]; Std_Corr_Fx=[tot_corr_fx_sd; std_corr_fx_subj];
    Mean_Corr_Fz=[tot_corr_fz_mu; mean_corr_fz_subj]; Std_Corr_Fz=[tot_corr_fz_sd; std_corr_fz_subj];
    
    % Flex/Ext (기존)
    Mean_nRMSE_Fx_Flex=[tot_nrmse_fx_flx_mu; mean_nrmse_fx_flx_subj]; Std_nRMSE_Fx_Flex=[tot_nrmse_fx_flx_sd; std_nrmse_fx_flx_subj];
    Mean_nRMSE_Fz_Flex=[tot_nrmse_fz_flx_mu; mean_nrmse_fz_flx_subj]; Std_nRMSE_Fz_Flex=[tot_nrmse_fz_flx_sd; std_nrmse_fz_flx_subj];
    Mean_nRMSE_Fx_Ext=[tot_nrmse_fx_ext_mu; mean_nrmse_fx_ext_subj]; Std_nRMSE_Fx_Ext=[tot_nrmse_fx_ext_sd; std_nrmse_fx_ext_subj];
    Mean_nRMSE_Fz_Ext=[tot_nrmse_fz_ext_mu; mean_nrmse_fz_ext_subj]; Std_nRMSE_Fz_Ext=[tot_nrmse_fz_ext_sd; std_nrmse_fz_ext_subj];
    Mean_Corr_Fx_Flex=[tot_corr_fx_flx_mu; mean_corr_fx_flx_subj]; Std_Corr_Fx_Flex=[tot_corr_fx_flx_sd; std_corr_fx_flx_subj];
    Mean_Corr_Fz_Flex=[tot_corr_fz_flx_mu; mean_corr_fz_flx_subj]; Std_Corr_Fz_Flex=[tot_corr_fz_flx_sd; std_corr_fz_flx_subj];
    Mean_Corr_Fx_Ext=[tot_corr_fx_ext_mu; mean_corr_fx_ext_subj]; Std_Corr_Fx_Ext=[tot_corr_fx_ext_sd; std_corr_fx_ext_subj];
    Mean_Corr_Fz_Ext=[tot_corr_fz_ext_mu; mean_corr_fz_ext_subj]; Std_Corr_Fz_Ext=[tot_corr_fz_ext_sd; std_corr_fz_ext_subj];
    
    % Ipsi/Contra (기존)
    Mean_nRMSE_Fx_Ipsi=[tot_nrmse_fx_ipsi_mu; mean_nrmse_fx_ipsi_subj]; Std_nRMSE_Fx_Ipsi=[tot_nrmse_fx_ipsi_sd; std_nrmse_fx_ipsi_subj];
    Mean_nRMSE_Fz_Ipsi=[tot_nrmse_fz_ipsi_mu; mean_nrmse_fz_ipsi_subj]; Std_nRMSE_Fz_Ipsi=[tot_nrmse_fz_ipsi_sd; std_nrmse_fz_ipsi_subj];
    Mean_nRMSE_Fx_Contra=[tot_nrmse_fx_contra_mu; mean_nrmse_fx_contra_subj]; Std_nRMSE_Fx_Contra=[tot_nrmse_fx_contra_sd; std_nrmse_fx_contra_subj];
    Mean_nRMSE_Fz_Contra=[tot_nrmse_fz_contra_mu; mean_nrmse_fz_contra_subj]; Std_nRMSE_Fz_Contra=[tot_nrmse_fz_contra_sd; std_nrmse_fz_contra_subj];
    Mean_Corr_Fx_Ipsi=[tot_corr_fx_ipsi_mu; mean_corr_fx_ipsi_subj]; Std_Corr_Fx_Ipsi=[tot_corr_fx_ipsi_sd; std_corr_fx_ipsi_subj];
    Mean_Corr_Fz_Ipsi=[tot_corr_fz_ipsi_mu; mean_corr_fz_ipsi_subj]; Std_Corr_Fz_Ipsi=[tot_corr_fz_ipsi_sd; std_corr_fz_ipsi_subj];
    Mean_Corr_Fx_Contra=[tot_corr_fx_contra_mu; mean_corr_fx_contra_subj]; Std_Corr_Fx_Contra=[tot_corr_fx_contra_sd; std_corr_fx_contra_subj];
    Mean_Corr_Fz_Contra=[tot_corr_fz_contra_mu; mean_corr_fz_contra_subj]; Std_Corr_Fz_Contra=[tot_corr_fz_contra_sd; std_corr_fz_contra_subj];
    
    % ===== [NEW] errTable에 추가할 Peak/Integral Error 열 =====
    Mean_Fx_Pos_Peak_Err  = [tot_fx_pp_err_mu; mean_fx_pp_err_subj]; Std_Fx_Pos_Peak_Err  = [tot_fx_pp_err_sd; std_fx_pp_err_subj];
    Mean_Fx_Neg_Peak_Err  = [tot_fx_np_err_mu; mean_fx_np_err_subj]; Std_Fx_Neg_Peak_Err  = [tot_fx_np_err_sd; std_fx_np_err_subj];
    Mean_Fz_Pos_Peak_Err  = [tot_fz_pp_err_mu; mean_fz_pp_err_subj]; Std_Fz_Pos_Peak_Err  = [tot_fz_pp_err_sd; std_fz_pp_err_subj];
    Mean_Fz_Neg_Peak_Err  = [tot_fz_np_err_mu; mean_fz_np_err_subj]; Std_Fz_Neg_Peak_Err  = [tot_fz_np_err_sd; std_fz_np_err_subj];
    Mean_Fx_Pos_Integral_Err = [tot_fx_pi_err_mu; mean_fx_pi_err_subj]; Std_Fx_Pos_Integral_Err = [tot_fx_pi_err_sd; std_fx_pi_err_subj];
    Mean_Fx_Neg_Integral_Err = [tot_fx_ni_err_mu; mean_fx_ni_err_subj]; Std_Fx_Neg_Integral_Err = [tot_fx_ni_err_sd; std_fx_ni_err_subj];
    Mean_Fz_Pos_Integral_Err = [tot_fz_pi_err_mu; mean_fz_pi_err_subj]; Std_Fz_Pos_Integral_Err = [tot_fz_pi_err_sd; std_fz_pi_err_subj];
    Mean_Fz_Neg_Integral_Err = [tot_fz_ni_err_mu; mean_fz_ni_err_subj]; Std_Fz_Neg_Integral_Err = [tot_fz_ni_err_sd; std_fz_ni_err_subj];
    
    % ===== [NEW] trueTable 열 =====
    Mean_Fx_Pos_Peak_True = [tot_fx_pp_true_mu; mean_fx_pp_true_subj]; Std_Fx_Pos_Peak_True = [tot_fx_pp_true_sd; std_fx_pp_true_subj];
    Mean_Fx_Pos_Peak_Pred = [tot_fx_pp_pred_mu; mean_fx_pp_pred_subj]; Std_Fx_Pos_Peak_Pred = [tot_fx_pp_pred_sd; std_fx_pp_pred_subj];
    Mean_Fx_Neg_Peak_True = [tot_fx_np_true_mu; mean_fx_np_true_subj]; Std_Fx_Neg_Peak_True = [tot_fx_np_true_sd; std_fx_np_true_subj];
    Mean_Fx_Neg_Peak_Pred = [tot_fx_np_pred_mu; mean_fx_np_pred_subj]; Std_Fx_Neg_Peak_Pred = [tot_fx_np_pred_sd; std_fx_np_pred_subj];
    Mean_Fz_Pos_Peak_True = [tot_fz_pp_true_mu; mean_fz_pp_true_subj]; Std_Fz_Pos_Peak_True = [tot_fz_pp_true_sd; std_fz_pp_true_subj];
    Mean_Fz_Pos_Peak_Pred = [tot_fz_pp_pred_mu; mean_fz_pp_pred_subj]; Std_Fz_Pos_Peak_Pred = [tot_fz_pp_pred_sd; std_fz_pp_pred_subj];
    Mean_Fz_Neg_Peak_True = [tot_fz_np_true_mu; mean_fz_np_true_subj]; Std_Fz_Neg_Peak_True = [tot_fz_np_true_sd; std_fz_np_true_subj];
    Mean_Fz_Neg_Peak_Pred = [tot_fz_np_pred_mu; mean_fz_np_pred_subj]; Std_Fz_Neg_Peak_Pred = [tot_fz_np_pred_sd; std_fz_np_pred_subj];
    
    Mean_Fx_Pos_Integral_True = [tot_fx_pi_true_mu; mean_fx_pi_true_subj]; Std_Fx_Pos_Integral_True = [tot_fx_pi_true_sd; std_fx_pi_true_subj];
    Mean_Fx_Pos_Integral_Pred = [tot_fx_pi_pred_mu; mean_fx_pi_pred_subj]; Std_Fx_Pos_Integral_Pred = [tot_fx_pi_pred_sd; std_fx_pi_pred_subj];
    Mean_Fx_Neg_Integral_True = [tot_fx_ni_true_mu; mean_fx_ni_true_subj]; Std_Fx_Neg_Integral_True = [tot_fx_ni_true_sd; std_fx_ni_true_subj];
    Mean_Fx_Neg_Integral_Pred = [tot_fx_ni_pred_mu; mean_fx_ni_pred_subj]; Std_Fx_Neg_Integral_Pred = [tot_fx_ni_pred_sd; std_fx_ni_pred_subj];
    Mean_Fz_Pos_Integral_True = [tot_fz_pi_true_mu; mean_fz_pi_true_subj]; Std_Fz_Pos_Integral_True = [tot_fz_pi_true_sd; std_fz_pi_true_subj];
    Mean_Fz_Pos_Integral_Pred = [tot_fz_pi_pred_mu; mean_fz_pi_pred_subj]; Std_Fz_Pos_Integral_Pred = [tot_fz_pi_pred_sd; std_fz_pi_pred_subj];
    Mean_Fz_Neg_Integral_True = [tot_fz_ni_true_mu; mean_fz_ni_true_subj]; Std_Fz_Neg_Integral_True = [tot_fz_ni_true_sd; std_fz_ni_true_subj];
    Mean_Fz_Neg_Integral_Pred = [tot_fz_ni_pred_mu; mean_fz_ni_pred_subj]; Std_Fz_Neg_Integral_Pred = [tot_fz_ni_pred_sd; std_fz_ni_pred_subj];
    % ===== [END NEW] =====
    
    % --- errTable: 기존 열 + NEW 열 추가 ---
    errTable = table(RowLabel, ...
        Mean_nRMSE_all, Std_nRMSE_all, ...
        Mean_nRMSE_Fx, Std_nRMSE_Fx, Mean_nRMSE_Fz, Std_nRMSE_Fz, ...
        Mean_Corr_all, Std_Corr_all, ...
        Mean_Corr_Fx, Std_Corr_Fx, Mean_Corr_Fz, Std_Corr_Fz, ...
        Mean_nRMSE_Fx_Flex, Std_nRMSE_Fx_Flex, Mean_nRMSE_Fz_Flex, Std_nRMSE_Fz_Flex, ...
        Mean_nRMSE_Fx_Ext, Std_nRMSE_Fx_Ext, Mean_nRMSE_Fz_Ext, Std_nRMSE_Fz_Ext, ...
        Mean_Corr_Fx_Flex, Std_Corr_Fx_Flex, Mean_Corr_Fz_Flex, Std_Corr_Fz_Flex, ...
        Mean_Corr_Fx_Ext, Std_Corr_Fx_Ext, Mean_Corr_Fz_Ext, Std_Corr_Fz_Ext, ...
        Mean_nRMSE_Fx_Ipsi, Std_nRMSE_Fx_Ipsi, Mean_nRMSE_Fz_Ipsi, Std_nRMSE_Fz_Ipsi, ...
        Mean_nRMSE_Fx_Contra, Std_nRMSE_Fx_Contra, Mean_nRMSE_Fz_Contra, Std_nRMSE_Fz_Contra, ...
        Mean_Corr_Fx_Ipsi, Std_Corr_Fx_Ipsi, Mean_Corr_Fz_Ipsi, Std_Corr_Fz_Ipsi, ...
        Mean_Corr_Fx_Contra, Std_Corr_Fx_Contra, Mean_Corr_Fz_Contra, Std_Corr_Fz_Contra, ...
        ... % ===== [NEW] Peak & Integral Error 열 =====
        Mean_Fx_Pos_Peak_Err, Std_Fx_Pos_Peak_Err, ...
        Mean_Fx_Neg_Peak_Err, Std_Fx_Neg_Peak_Err, ...
        Mean_Fz_Pos_Peak_Err, Std_Fz_Pos_Peak_Err, ...
        Mean_Fz_Neg_Peak_Err, Std_Fz_Neg_Peak_Err, ...
        Mean_Fx_Pos_Integral_Err, Std_Fx_Pos_Integral_Err, ...
        Mean_Fx_Neg_Integral_Err, Std_Fx_Neg_Integral_Err, ...
        Mean_Fz_Pos_Integral_Err, Std_Fz_Pos_Integral_Err, ...
        Mean_Fz_Neg_Integral_Err, Std_Fz_Neg_Integral_Err);
    
    % --- [NEW] trueTable 생성 ---
    trueTable = table(RowLabel, ...
        Mean_Fx_Pos_Peak_True, Std_Fx_Pos_Peak_True, ...
        Mean_Fx_Pos_Peak_Pred, Std_Fx_Pos_Peak_Pred, ...
        Mean_Fx_Neg_Peak_True, Std_Fx_Neg_Peak_True, ...
        Mean_Fx_Neg_Peak_Pred, Std_Fx_Neg_Peak_Pred, ...
        Mean_Fz_Pos_Peak_True, Std_Fz_Pos_Peak_True, ...
        Mean_Fz_Pos_Peak_Pred, Std_Fz_Pos_Peak_Pred, ...
        Mean_Fz_Neg_Peak_True, Std_Fz_Neg_Peak_True, ...
        Mean_Fz_Neg_Peak_Pred, Std_Fz_Neg_Peak_Pred, ...
        Mean_Fx_Pos_Integral_True, Std_Fx_Pos_Integral_True, ...
        Mean_Fx_Pos_Integral_Pred, Std_Fx_Pos_Integral_Pred, ...
        Mean_Fx_Neg_Integral_True, Std_Fx_Neg_Integral_True, ...
        Mean_Fx_Neg_Integral_Pred, Std_Fx_Neg_Integral_Pred, ...
        Mean_Fz_Pos_Integral_True, Std_Fz_Pos_Integral_True, ...
        Mean_Fz_Pos_Integral_Pred, Std_Fz_Pos_Integral_Pred, ...
        Mean_Fz_Neg_Integral_True, Std_Fz_Neg_Integral_True, ...
        Mean_Fz_Neg_Integral_Pred, Std_Fz_Neg_Integral_Pred);
        
    % -------------------------------------------------
    % 2-5. Save All (Struct & Tables)
    % -------------------------------------------------
    assignin('base', struct_var_name, plot_data);
    save(full_save_path_struct, struct_var_name);
    fprintf('  >> Struct 저장: %s\n', mat_filename_struct);
    
    % errTable 저장 (기존)
    errorVarName_str = "errorTable_" + base_name;
    errorVarName_char = char(errorVarName_str);
    assignin('base', errorVarName_char, errTable);
    error_mat_file = fullfile(current_folder, errorVarName_str + ".mat");
    save(char(error_mat_file), errorVarName_char);
    fprintf('  >> errTable 저장 완료: %s\n', errorVarName_str+".mat");
    
    % ===== [NEW] trueTable 저장 =====
    trueVarName_str = "trueTable_" + base_name;
    trueVarName_char = char(trueVarName_str);
    assignin('base', trueVarName_char, trueTable);
    true_mat_file = fullfile(current_folder, trueVarName_str + ".mat");
    save(char(true_mat_file), trueVarName_char);
    fprintf('  >> trueTable 저장 완료: %s\n', trueVarName_str+".mat");
    % ===== [END NEW] =====
    
    evalin('base', sprintf('clear %s %s %s', struct_var_name, errorVarName_char, trueVarName_char));
end
fprintf('\n모든 CSV 파일에 대한 변환 및 테이블 생성이 완료되었습니다.\n');