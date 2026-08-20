% ================================================================
%  [revision] CoM 오프셋 섭동 데이터 생성 — 세그먼트별 분리
%
%  리뷰어 지적
%     "데이터셋 A-B 간 체계적 마커 부착 편향(몸통 오프셋)이 최종 GRF 오차로
%      전파되는 정도가 정량화되지 않았음. 간단한 민감도 추정이 도움이 될 것."
%
%  설계
%     jinsung_Modeling_Third.m 의 오프셋에 배율을 곱해 stride_modeling 을 재생성한다.
%     학습된 모델은 건드리지 않고 추론 입력만 바꿔 민감도를 본다.
%
%     ※ 분절마다 민감도가 다르므로 **하나씩 따로** 흔든다.
%          SEG='we'  전완  (wrist-elbow)
%          SEG='es'  상완  (elbow-shoulder)
%          SEG='sc'  몸통  (shoulder-sacrum)   <- 리뷰어의 관심사
%          SEG='all' 세 분절 동시 (참고용)
%
%     ※ 원 스크립트는 INCLUDE_SC=false 로 되어 있어 몸통 항이 계산조차 되지 않는다.
%       (그 경우 저장되는 sacrum 값은 실제로는 어깨 가속도다.)
%       원 학습 데이터는 전 분절을 포함해 만들었으므로 여기서 INCLUDE_SC=true 로 강제한다.
%
%  Eq.(3): th_i_fd = th_i_pred + off_i.  off 에 Δ 를 더하면 그 분절의 기여 벡터가
%  정확히 Δ 만큼 회전한다 (rddx' = cosΔ·rddx − sinΔ·rddz).
%
%  사용법 (MATLAB) — 논문 Section III-B 에 보고한 설정
%     BASE_DIR='<experiment_integration 출력 폴더>';
%     OUT_DIR ='<eval_offset_perturbation.py 의 PERTURB_DIR>';
%     SEG='sc'; DELTA_LIST=[-30,-15,15,30]; run('run_offset_perturbation.m')
%     SEG='none';                           run('run_offset_perturbation.m')  % baseline
% ================================================================

if ~exist('SEG','var'),        SEG = 'sc'; end
if ~exist('FORCE_SC','var'),   FORCE_SC = true; end
if ~exist('BASE_DIR','var')
    % experiment_integration.m 이 만든 .mat 폴더로 바꿔 주십시오.
    BASE_DIR = fullfile('..', '..', '..', 'data');
end
if ~exist('OUT_DIR','var')
    % eval_offset_perturbation.py 의 PERTURB_DIR 과 같은 폴더여야 합니다.
    OUT_DIR = fullfile('..', '..', '..', 'data', 'perturb');
end
if ~exist(OUT_DIR,'dir'), mkdir(OUT_DIR); end

fprintf('=== 오프셋 섭동 생성 ===\n');
fprintf('SEG=%s  FORCE_SC=%d\n', SEG, FORCE_SC);

% ---------------- 입력 로드 (한 번만) ----------------
if ~(exist('stride_kinematics_arm','var') && exist('stride_kinematics_arm_vector','var'))
    kin_file = fullfile(BASE_DIR, 'merged_stride_kinematics_arm_gyro.mat');
    vec_file = fullfile(BASE_DIR, 'merged_stride_kinematics_arm_vector_gyro.mat');
    fprintf('[load] %s\n', kin_file);  Lk = load(kin_file);
    fprintf('[load] %s\n', vec_file);  Lv = load(vec_file);
    fn = fieldnames(Lk);  stride_kinematics_arm        = Lk.(fn{1});
    fn = fieldnames(Lv);  stride_kinematics_arm_vector = Lv.(fn{1});
    clear Lk Lv
    fprintf('[ok] 로드 완료\n');
end

src = fileread(fullfile(BASE_DIR, 'jinsung_Modeling_Third.m'));

% 원 스크립트는 한글 주석 인코딩이 깨져 있어 ASCII 코드 라인만 앵커로 쓴다.
anchor = 'if APPLY_OFFSET_TO_FORWARD';
assert(count(src, anchor) == 1, '앵커가 유일하지 않습니다.');

% 몸통(SC) 항을 살린다. 원 학습 데이터가 전 분절 포함이므로 필수.
if FORCE_SC
    n_before = count(src, 'INCLUDE_SC = false;');
    src = strrep(src, 'INCLUDE_SC = false;', 'INCLUDE_SC = true;');
    fprintf('[patch] INCLUDE_SC false->true (%d 곳)\n', n_before);
end

% 원 학습 데이터의 생성 설정을 재현하기 위한 오버라이드
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

% 가산(additive) 섭동만 사용한다:  off_i <- off_i + deg2rad(DELTA)
%   마커 부착 편향은 모든 stride 에 동일하게 작용하는 상수이므로,
%   실측 편차(약 15도)를 문자 그대로 재현하려면 가산이 맞다.
IND = repmat(' ', 1, 16);
op_we = 'off_we = off_we + deg2rad(OFFSET_DELTA);';
op_es = 'if exist(''off_es'',''var''), off_es = off_es + deg2rad(OFFSET_DELTA); end';
op_sc = 'if exist(''off_sc'',''var''), off_sc = off_sc + deg2rad(OFFSET_DELTA); end';
switch lower(SEG)
    case 'none',  lines = {};                                    % baseline
    case 'we',    lines = {op_we};
    case 'es',    lines = {op_es};
    case 'sc',    lines = {op_sc};
    case 'all',   lines = {op_we, op_es, op_sc};
    otherwise, error('SEG=%s 는 알 수 없습니다.', SEG);
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

    % ※ 파일명에 '-' 가 들어가면 run() 이 수식으로 해석해 실패한다. 부호는 문자로 표기.
    if p < 0, tsgn = 'm'; else, tsgn = 'p'; end
    tmp = fullfile(tempdir, sprintf('mod3_%s_add_%s%d.m', ...
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

fprintf('\n=== 완료 ===\n');
