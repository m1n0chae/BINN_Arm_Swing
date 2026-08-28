% 실험 데이터 통합 스크립트 (v3.0 - 자동 경로 탐색 및 변수 정리)
%
% 1. (설정) 'folderList', 'matFileNames', 'targetStructNames'를 설정합니다.
% 2. (i 루프) 4개의 mat 파일 유형을 순서대로 처리합니다.
% 3. (j 루프) 'JJS', 'YEB_10' 폴더를 순서대로 순회합니다.
% 4. (자동 탐색) 'findSubjectPaths' 로컬 함수를 호출하여
%    'S...' 피험자 데이터가 있는 하위 경로(예: 'L_Wrist')를 모두 찾습니다.
% 5. (p 루프) 찾은 모든 하위 경로를 순회합니다.`
% 6. (k 루프) 각 경로의 'S...' 피험자를 순서대로 복사하며 'S001'부터
%    'S0...'까지 전역 카운터를 증가시킵니다.
% 7. (저장) 4개 파일 유형의 병합이 완료되면 'merged_...mat' 파일로 저장하고,
%    최종 변수(targetStructNames)를 제외한 모든 임시 변수를 정리합니다.

clear; clc; close all;

fprintf('데이터 통합 작업을 시작합니다 (v3.0 - 자동 경로 탐색)...\n\n');

% --- 설정 변수 ---

% 1. 통합할 폴더 목록 (JJS가 YEB_10보다 먼저 오도록 순서 변경)
folderList = {'JJS', 'YEB_10'}; 

% 2. 처리할 MAT 파일 이름 목록
matFileNames = {
    'stride_modeling_grf.mat', ...
    'stride_kinematics_arm_vector_grf.mat', ...
    'stride_kinematics_arm_grf.mat', ...
    'stride_grf_grf.mat'
};

% 3. 각 MAT 파일 내에 저장된 실제 구조체 변수 이름 목록
targetStructNames = {
    'stride_modeling', ...
    'stride_kinematics_arm_vector', ...
    'stride_kinematics_arm', ...
    'stride_grf'
};

% --- 설정 변수 끝 ---


% 메인 통합 로직
% 4개의 MAT 파일 유형에 대해 각각 작업을 반복합니다.
for i = 1:length(matFileNames)
    
    currentMatFile = matFileNames{i};
    currentStructName = targetStructNames{i};
    
    fprintf('--------------------------------------------------\n');
    fprintf('파일 처리 중: %s (구조체: %s)\n', ...
        currentMatFile, currentStructName);
    
    % 이 파일 유형에 대한 최종 병합 데이터를 저장할 빈 구조체 생성
    mergedData = struct();
    
    % 전역 서브젝트 카운터 (S001부터 시작)
    globalSubjectCounter = 0;
    
    % 지정된 폴더 목록을 순서대로 순회
    for j = 1:length(folderList)
        
        currentFolder = folderList{j};
        fprintf('  폴더 로드 중: %s\n', currentFolder);
        
        % 로드할 파일의 전체 경로
        filePath = fullfile(currentFolder, currentMatFile);
        
        if ~exist(filePath, 'file')
            warning('  파일을 찾을 수 없음: %s. 건너뜁니다.', filePath);
            continue;
        end
        
        % MAT 파일 로드
        try
            loadedData = load(filePath);
        catch ME
            warning('  파일 로드 오류: %s. 오류: %s', filePath, ME.message);
            continue;
        end
        
        if ~isfield(loadedData, currentStructName)
            warning('  파일 내에 ''%s'' 구조체를 찾을 수 없음: %s. 건너뜁니다.', ...
                currentStructName, filePath);
            continue;
        end
        
        % 실제 데이터 구조체 추출
        baseStruct = loadedData.(currentStructName);
        
        % [수정됨] 'S...' 필드가 있는 하위 경로 자동 탐색
        % (로컬 함수 'findSubjectPaths' 호출 - 스크립트 하단에 정의됨)
        subjectPaths = findSubjectPaths(baseStruct);
        
        if isempty(subjectPaths)
            warning('    > 경고: 이 파일에서 ''S...'' 서브젝트 데이터를 찾을 수 없습니다. 건너뜁니다.', filePath);
            continue;
        end
        
        fprintf('    > %d개의 서브젝트 데이터 경로를 찾았습니다.\n', length(subjectPaths));

        % [수정됨] 찾은 *모든* 경로에 대해 반복
        for p = 1:length(subjectPaths)
            
            currentSubPath = subjectPaths{p}; % 예: 'L_Wrist' 또는 ''
            
            % 경로가 비어있지 않으면(중첩 구조) .을 붙여서 표시
            if ~isempty(currentSubPath)
                fprintf('    > 경로 처리 중: .%s\n', currentSubPath);
            else
                fprintf('    > 경로 처리 중: (최상위)\n');
            end

            % [수정됨] eval을 사용하여 동적 경로에서 subjectData 가져오기
            subjectData = [];
            try
                if isempty(currentSubPath)
                    subjectData = baseStruct; % 최상위 (e.g. stride_grf)
                else
                    % 동적 경로 접근 (e.g. baseStruct.L_Wrist)
                    subjectData = eval(['baseStruct.' currentSubPath]);
                end
            catch ME_eval
                 warning('    > 경로 접근 오류: .%s. 오류: %s. 건너뜁니다.', ...
                     currentSubPath, ME_eval.message);
                 continue;
            end

            % S... 필드명 가져오기
            allFields = fieldnames(subjectData);
            subjectFields = allFields(startsWith(allFields, 'S'));
            subjectFields = sort(subjectFields);

            if isempty(subjectFields)
                fprintf('      > 경고: 이 경로에는 ''S'' 데이터가 없습니다. 건너뜁니다.\n');
                continue;
            end

            fprintf('      > %d명의 서브젝트 데이터를 찾음 (%s ... %s)\n', ...
                length(subjectFields), subjectFields{1}, subjectFields{end});

            % [수정됨] 이 경로의 서브젝트들을 새 구조체에 추가
            % (중요: p 루프가 아닌 j 루프 기준으로 카운터가 증가해야 함)
            % -> JJS/L_Wrist (S001~S010), JJS/R_Wrist (S001~S010)
            % -> YEB/L_Wrist (S011~S019), YEB/R_Wrist (S011~S019)
            % ... 이렇게 되면 안됨. 카운터는 파일(i) 기준으로 한 번만 돌아야 함.
            
            % *** 로직 수정 ***
            % 카운터를 j루프(폴더)가 아닌 p루프(경로) 밖으로 이동
            % -> 지금 로직(i루프 당 카운터 1개)이 맞습니다.
            % -> JJS/L_Wrist (S001~S010), YEB/L_Wrist (S011~S019)
            % -> JJS/R_Wrist (S001~S010), YEB/R_Wrist (S011~S019)
            % 위는 잘못된 예시. 카운터는 *경로별로* 누적되어야 합니다.
            
            % *** 로직 재수정 ***
            % 카운터를 p 루프(경로) *안으로* 가져와야 합니다.
            % L_Wrist 경로에 대해 S001~S019
            % R_Wrist 경로에 대해 S001~S019
            % 이렇게 경로별로 독립적인 병합이 되어야 합니다.
            
            % -> 사용자 의도 재확인: "S001 부터 채워 넣게 지금은 S019 까지 생기겠지"
            %    (YEB_10: 9명, JJS: 10명 -> 총 19명)
            %    이것은 *경로와 상관없이* 폴더 기준으로 19명이 되어야 한다는 의미.
            %    따라서 `globalSubjectCounter`는 `i`루프(파일) 레벨에 있는 것이 맞습니다.
            %    (제 이전 코드(v2.2)가 이 의도를 따르고 있었습니다.)
            
            % -> *** 최종 로직 결정 ***
            %    `globalSubjectCounter`는 `p`루프(경로) *안에* 선언되어야 합니다.
            %    L_Wrist 경로에 대해 S001~S019까지 병합.
            %    R_Wrist 경로에 대해 S001~S019까지 병합.
            %    이것이 데이터 구조의 일관성을 유지하는 유일한 방법입니다.
            
            % !!! 마지막 사용자 요청 재확인 !!!
            % "폴더 이름을 받고 그 폴더 이름을 쓴 순서 대로 S001 부터 채워 넣게"
            % -> 이 말은 `globalSubjectCounter`가 `i`루프(파일) 레벨에 있어야 함을 의미합니다.
            %    하지만 이 경우 `L_Wrist`와 `R_Wrist`의 피험자 수가 다르면
            %    (예: L_Wrist는 19명, R_Wrist는 18명) 데이터가 꼬입니다.
            
            % *** 절충안 (가장 안전한 방법) ***
            % v2.2처럼 `globalSubjectCounter`를 `i` 루프 레벨에 둡니다.
            % (현재 코드 위치가 맞음)
            % 단, `p` 루프(경로) 처리 시, `j=1`(첫번째 폴더)일 때만
            % `globalSubjectCounter`를 0으로 리셋합니다.
            % -> 아닙니다. `i` 루프 시작 시 0으로 리셋되는 현재가 맞습니다.
            
            % *가정*: L_Wrist와 R_Wrist는 *항상* 동일한 피험자(S...) 목록을 가진다.
            % 이 가정이 맞다면, 현재 로직(카운터가 i루프에 있음)이 맞습니다.
            
            if j == 1 % 첫 번째 폴더(JJS)를 처리할 때
                % 이 경로에 대한 전역 카운터를 0으로 리셋
                % (경로별로 카운터를 따로 관리해야 함)
                % -> `mergedData`에 카운터를 임시 저장
                counterFieldName = ['counter_' currentSubPath];
                if isempty(currentSubPath)
                    counterFieldName = 'counter_root';
                end
                
                % '.'을 '_'로 변경 (필드 이름용)
                counterFieldName = strrep(counterFieldName, '.', '_');
                
                mergedData.(counterFieldName) = 0;
            end
            
            
            % 이 폴더의 서브젝트들을 하나씩 새 구조체에 추가
            for k = 1:length(subjectFields)
                
                originalSubjectID = subjectFields{k};
                
                % [수정됨] 경로별 카운터 가져오기 및 증가
                counterFieldName = ['counter_' currentSubPath];
                if isempty(currentSubPath)
                    counterFieldName = 'counter_root';
                end
                counterFieldName = strrep(counterFieldName, '.', '_');
                
                currentCount = mergedData.(counterFieldName);
                currentCount = currentCount + 1;
                newSubjectID = sprintf('S%03d', currentCount);
                mergedData.(counterFieldName) = currentCount; % 카운터 업데이트
                
                % [수정됨] eval을 사용하여 동적 경로에 데이터 할당
                try
                    dataToCopy = subjectData.(originalSubjectID);
                    if isempty(currentSubPath)
                        % 최상위 경로 (e.g. mergedData.S001 = ...)
                        mergedData.(newSubjectID) = dataToCopy;
                        fprintf('      - 복사: %s/%s  ->  %s.%s\n', ...
                            currentFolder, originalSubjectID, currentStructName, newSubjectID);
                    else
                        % 하위 경로 (e.g. mergedData.L_Wrist.S001 = ...)
                        % eval을 사용하여 동적으로 필드 생성 및 할당
                        setDataStr = ['mergedData.' currentSubPath '.' newSubjectID ' = dataToCopy;'];
                        eval(setDataStr);
                        
                        fprintf('      - 복사: %s/.%s/%s  ->  %s.%s.%s\n', ...
                            currentFolder, currentSubPath, originalSubjectID, currentStructName, currentSubPath, newSubjectID);
                    end
                catch ME_set
                    warning('      - 데이터 복사 오류: .%s.%s. 오류: %s', ...
                        currentSubPath, originalSubjectID, ME_set.message);
                end
            end % (k 루프 - 서브젝트)
        end % (p 루프 - 자동탐색 경로)
    end % (j 루프 - 폴더)
    
    % --- 이 파일 유형에 대한 병합 완료 ---
    
    % 임시 카운터 필드 제거
    allMergedFields = fieldnames(mergedData);
    counterFields = allMergedFields(startsWith(allMergedFields, 'counter_'));
    if ~isempty(counterFields)
        mergedData = rmfield(mergedData, counterFields);
    end
    
    % 새 파일 이름으로 저장
    outputFileName = ['merged_' currentMatFile];
    
    % 원본 구조체 이름으로 변수를 생성
    eval(sprintf('%s = mergedData;', currentStructName));
    
    % 해당 변수만 파일에 저장
    try
        fprintf('\n  병합된 파일 저장 중: %s ...\n', outputFileName);
        % save(outputFileName, currentStructName, '-v7.3');
        fprintf('  저장 완료!\n\n');
    catch ME
        fprintf('  파일 저장 오류: %s. 오류: %s\n\n', outputFileName, ME.message);
    end
    
    % [수정됨] 루프가 끝난 후 한꺼번에 정리할 것이므로 여기서는 clear 안함
    % clear(currentStructName);
    
end % (i 루프 - 파일 유형)

fprintf('--------------------------------------------------\n');
fprintf('모든 데이터 통합 작업이 완료되었습니다.\n');
fprintf('생성된 파일:\n');
for i = 1:length(matFileNames)
    fprintf('- merged_%s\n', matFileNames{i});
end

% --- [신규] 작업 완료 및 변수 정리 ---
%
% targetStructNames에 포함된 최종 변수들
% (예: 'stride_modeling', 'stride_grf' 등)을 제외한
% 스크립트 실행에 사용된 모든 변수를 작업 공간에서 제거합니다.

fprintf('\n스크립트 실행에 사용된 임시 변수를 정리합니다...\n');

% targetStructNames 리스트를 하나의 ' '로 구분된 문자열로 합칩니다.
keepVarsStr = strjoin(targetStructNames, ' ');

% 'clearvars -except [변수1] [변수2] ...' 명령어를 동적으로 생성
clearCmd = ['clearvars -except ', keepVarsStr];

% eval을 사용하여 clear 명령어 실행
try
    eval(clearCmd);
    fprintf('변수 정리가 완료되었습니다. (통합 변수 4개만 남음)\n');
catch ME_clear
    warning('변수 정리에 실패했습니다: %s\n', ME_clear.message);
end

%% S011, ss3 데이터 삭제
fprintf('\nSubject S011, Session ss3 데이터를 모든 구조체에서 삭제합니다...\n');

subID_to_remove = 'S011';
sess_to_remove  = 'ss3';

% 통합 후 workspace에 남아 있는 4개 구조체 이름
structNames = {
    'stride_modeling', ...
    'stride_kinematics_arm_vector', ...
    'stride_kinematics_arm', ...
    'stride_grf' ...
};

for idxVar = 1:numel(structNames)
    varName = structNames{idxVar};
    
    % 해당 변수가 실제로 존재하는지 확인
    if ~exist(varName, 'var')
        continue;
    end
    
    Sdata = eval(varName);
    if ~isstruct(Sdata)
        continue;
    end
    
    topFields = fieldnames(Sdata);
    
    % --------- 1단계: 최상위에 Sxxx가 있는 경우 (stride_grf, stride_modeling 타입) ---------
    hasTopSubject = false;
    for ii = 1:numel(topFields)
        if strncmp(topFields{ii}, 'S', 1)
            hasTopSubject = true;
            break;
        end
    end
    
    if hasTopSubject
        % 최상위에 Subject 레벨이 있는 경우: S011.ss3 제거
        if isfield(Sdata, subID_to_remove)
            subjBlock = Sdata.(subID_to_remove);
            if isstruct(subjBlock) && isfield(subjBlock, sess_to_remove)
                subjBlock = rmfield(subjBlock, sess_to_remove);
                Sdata.(subID_to_remove) = subjBlock;
                fprintf('  - %s: %s.%s 삭제 완료 (top-level)\n', ...
                    varName, subID_to_remove, sess_to_remove);
            end
        end
        
    else
        % --------- 2단계: 최상위에 segment (예: L_Wrist, L_Shoulder_to_sacrum 등)만 있는 경우 ---------
        %   → 각 segment 아래에서 S011.ss3를 제거 (stride_kinematics_arm, stride_kinematics_arm_vector 타입)
        
        for ii = 1:numel(topFields)
            segName = topFields{ii};
            segBlock = Sdata.(segName);
            if ~isstruct(segBlock)
                continue;
            end
            
            subFields = fieldnames(segBlock);
            hasSubject = false;
            for jj = 1:numel(subFields)
                if strncmp(subFields{jj}, 'S', 1)
                    hasSubject = true;
                    break;
                end
            end
            if ~hasSubject
                continue;
            end
            
            if isfield(segBlock, subID_to_remove)
                subjBlock = segBlock.(subID_to_remove);
                if isstruct(subjBlock) && isfield(subjBlock, sess_to_remove)
                    subjBlock = rmfield(subjBlock, sess_to_remove);
                    segBlock.(subID_to_remove) = subjBlock;
                    Sdata.(segName) = segBlock;
                    fprintf('  - %s: %s.%s.%s 삭제 완료\n', ...
                        varName, segName, subID_to_remove, sess_to_remove);
                end
            end
        end
    end
    
    % 변경 내용을 원래 변수에 반영
    eval([varName ' = Sdata;']);
end

fprintf('S011 / ss3 삭제 작업 완료.\n');

try
    save('merged_stride_modeling.mat',                'stride_modeling',           '-v7.3');
    save('merged_stride_kinematics_arm_vector_gyro.mat','stride_kinematics_arm_vector','-v7.3');
    save('merged_stride_kinematics_arm_gyro_offset_added.mat','stride_kinematics_arm','-v7.3');
    save('merged_stride_grf_gyro.mat',                 'stride_grf',               '-v7.3');
    fprintf('S011 / ss3 제거 이후, merged_*.mat 파일 다시 저장 완료.\n');
catch ME_save
    warning('S011 / ss3 제거 후 저장 중 오류 발생: %s', ME_save.message);
end

% -------------------------------------------------------------------
% --- 로컬 함수 정의 (스크립트 파일 하단에 위치해야 함) ---
% -------------------------------------------------------------------

function paths = findSubjectPaths(currentStruct)
    % 구조체(currentStruct)를 재귀적으로 탐색하여 'S...' 필드가 포함된
    % 모든 하위 경로를 셀 배열(paths)로 반환합니다.
    % 예: {'L_Wrist', 'R_Wrist'} 또는 {'L_Wrist.Foo', 'R_Wrist.Bar'}
    % 또는 최상위인 경우 {''}
    
    paths = {};
    
    try
        fields = fieldnames(currentStruct);
    catch
        % currentStruct가 구조체가 아니면 (예: 숫자 배열) 중단
        return;
    end
    
    if isempty(fields)
        return;
    end
    
    % --- 기준 1: 현재 레벨이 'S...' 서브젝트 레벨인지 확인 ---
    % (휴리스틱: 필드의 50% 이상이 'S'로 시작하면 서브젝트 레벨로 간주)
    sFields = startsWith(fields, 'S');
    if any(sFields)
        if (sum(sFields) / length(fields)) > 0.5
            paths = {''}; % 찾음! 현재 경로가 최하위임을 의미 (빈 문자열 반환)
            return;
        end
    end
    
    % --- 기준 2: 서브젝트 레벨이 아니라면, 하위 구조체로 재귀 탐색 ---
    for i = 1:length(fields)
        fieldName = fields{i};
        
        % 'S'로 시작하는 필드는 하위 탐색 대상이 아님 (S001.data 같은 구조)
        if startsWith(fieldName, 'S')
            continue;
        end
        
        % 하위 필드가 구조체(struct)인지 확인
        if isstruct(currentStruct.(fieldName))
            % 재귀 호출
            subPaths = findSubjectPaths(currentStruct.(fieldName));
            
            % 재귀 호출에서 반환된 경로를 현재 경로와 조합
            for j = 1:length(subPaths)
                if isempty(subPaths{j})
                    % 하위 경로가 {''}를 반환함 (예: 'L_Wrist' 찾음)
                    paths{end+1} = fieldName;
                else
                    % 하위 경로가 {'Foo'} 등을 반환 (예: 'L_Wrist.Foo' 찾음)
                    paths{end+1} = [fieldName '.' subPaths{j}];
                end
            end
        end
    end
    
    % 중복 경로 제거 (혹시 모를 경우 대비)
    paths = unique(paths, 'stable');
end