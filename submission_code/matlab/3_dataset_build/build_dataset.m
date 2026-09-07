% Experiment-data merging script (v3.0 - automatic path discovery and variable cleanup)
%
% 1. (setup)   Configure 'folderList', 'matFileNames', 'targetStructNames'.
% 2. (i loop)  Process the four .mat file types in order.
% 3. (j loop)  Iterate over the 'JJS' and 'YEB_10' folders in order.
% 4. (auto discovery) Call the local function 'findSubjectPaths' to find every
%    sub-path (e.g. 'L_Wrist') that contains 'S...' subject data.
% 5. (p loop)  Iterate over all discovered sub-paths.
% 6. (k loop)  Copy each path's 'S...' subjects in order, incrementing a global
%    counter so the merged struct is numbered from 'S001' upwards.
% 7. (save)    After the four file types are merged, save them as 'merged_...mat'
%    and clear every temporary variable except the final structs
%    (targetStructNames).

clear; clc; close all;

fprintf('Starting data merge (v3.0 - automatic path discovery)...\n\n');

% --- configuration ---

% 1. folders to merge (JJS deliberately ordered before YEB_10)
folderList = {'JJS', 'YEB_10'};

% 2. MAT file names to process
matFileNames = {
    'stride_modeling_grf.mat', ...
    'stride_kinematics_arm_vector_grf.mat', ...
    'stride_kinematics_arm_grf.mat', ...
    'stride_grf_grf.mat'
};

% 3. names of the struct variables stored inside each MAT file
targetStructNames = {
    'stride_modeling', ...
    'stride_kinematics_arm_vector', ...
    'stride_kinematics_arm', ...
    'stride_grf'
};

% --- end of configuration ---


% Main merge logic
% Repeat the work for each of the four MAT file types.
for i = 1:length(matFileNames)

    currentMatFile = matFileNames{i};
    currentStructName = targetStructNames{i};

    fprintf('--------------------------------------------------\n');
    fprintf('Processing file: %s (struct: %s)\n', ...
        currentMatFile, currentStructName);

    % empty struct that will hold the merged data for this file type
    mergedData = struct();

    % global subject counter (numbering starts at S001)
    globalSubjectCounter = 0;

    % iterate over the configured folders in order
    for j = 1:length(folderList)

        currentFolder = folderList{j};
        fprintf('  Loading folder: %s\n', currentFolder);

        % full path of the file to load
        filePath = fullfile(currentFolder, currentMatFile);

        if ~exist(filePath, 'file')
            warning('  File not found: %s. Skipping.', filePath);
            continue;
        end

        % load the MAT file
        try
            loadedData = load(filePath);
        catch ME
            warning('  File load error: %s. Error: %s', filePath, ME.message);
            continue;
        end

        if ~isfield(loadedData, currentStructName)
            warning('  Struct ''%s'' not found inside the file: %s. Skipping.', ...
                currentStructName, filePath);
            continue;
        end

        % extract the actual data struct
        baseStruct = loadedData.(currentStructName);

        % automatically discover the sub-paths that carry 'S...' fields
        % (local function 'findSubjectPaths', defined at the bottom of this script)
        subjectPaths = findSubjectPaths(baseStruct);

        if isempty(subjectPaths)
            warning('    > No ''S...'' subject data found in this file. Skipping.', filePath);
            continue;
        end

        fprintf('    > Found %d subject-data path(s).\n', length(subjectPaths));

        % iterate over *all* discovered paths
        for p = 1:length(subjectPaths)

            currentSubPath = subjectPaths{p}; % e.g. 'L_Wrist' or ''

            % non-empty paths (nested structs) are displayed with a leading dot
            if ~isempty(currentSubPath)
                fprintf('    > Processing path: .%s\n', currentSubPath);
            else
                fprintf('    > Processing path: (top level)\n');
            end

            % fetch subjectData from the dynamic path via eval
            subjectData = [];
            try
                if isempty(currentSubPath)
                    subjectData = baseStruct; % top level (e.g. stride_grf)
                else
                    % dynamic path access (e.g. baseStruct.L_Wrist)
                    subjectData = eval(['baseStruct.' currentSubPath]);
                end
            catch ME_eval
                 warning('    > Path access error: .%s. Error: %s. Skipping.', ...
                     currentSubPath, ME_eval.message);
                 continue;
            end

            % collect the S... field names
            allFields = fieldnames(subjectData);
            subjectFields = allFields(startsWith(allFields, 'S'));
            subjectFields = sort(subjectFields);

            if isempty(subjectFields)
                fprintf('      > Warning: no ''S'' data on this path. Skipping.\n');
                continue;
            end

            fprintf('      > Found %d subject(s) (%s ... %s)\n', ...
                length(subjectFields), subjectFields{1}, subjectFields{end});

            % Append this path's subjects to the new struct.
            %
            % Counter design: renumbering must run PER PATH but ACROSS folders,
            % i.e. for each path (L_Wrist, R_Wrist, ...) the subjects of the
            % first folder (JJS, 10 subjects) become S001-S010 and the second
            % folder (YEB_10, 9 subjects) continues as S011-S019. A per-path
            % counter is therefore kept inside mergedData and reset when the
            % first folder is processed. This assumes every path carries the
            % same subject list, which holds for these data.

            if j == 1 % when processing the first folder (JJS)
                % reset the per-path counter
                counterFieldName = ['counter_' currentSubPath];
                if isempty(currentSubPath)
                    counterFieldName = 'counter_root';
                end

                % replace '.' with '_' (field-name safe)
                counterFieldName = strrep(counterFieldName, '.', '_');

                mergedData.(counterFieldName) = 0;
            end


            % append this folder's subjects to the new struct one by one
            for k = 1:length(subjectFields)

                originalSubjectID = subjectFields{k};

                % fetch and increment the per-path counter
                counterFieldName = ['counter_' currentSubPath];
                if isempty(currentSubPath)
                    counterFieldName = 'counter_root';
                end
                counterFieldName = strrep(counterFieldName, '.', '_');

                currentCount = mergedData.(counterFieldName);
                currentCount = currentCount + 1;
                newSubjectID = sprintf('S%03d', currentCount);
                mergedData.(counterFieldName) = currentCount; % update the counter

                % assign the data on the dynamic path via eval
                try
                    dataToCopy = subjectData.(originalSubjectID);
                    if isempty(currentSubPath)
                        % top-level path (e.g. mergedData.S001 = ...)
                        mergedData.(newSubjectID) = dataToCopy;
                        fprintf('      - copied: %s/%s  ->  %s.%s\n', ...
                            currentFolder, originalSubjectID, currentStructName, newSubjectID);
                    else
                        % nested path (e.g. mergedData.L_Wrist.S001 = ...)
                        % dynamically create and assign the field via eval
                        setDataStr = ['mergedData.' currentSubPath '.' newSubjectID ' = dataToCopy;'];
                        eval(setDataStr);

                        fprintf('      - copied: %s/.%s/%s  ->  %s.%s.%s\n', ...
                            currentFolder, currentSubPath, originalSubjectID, currentStructName, currentSubPath, newSubjectID);
                    end
                catch ME_set
                    warning('      - data copy error: .%s.%s. Error: %s', ...
                        currentSubPath, originalSubjectID, ME_set.message);
                end
            end % (k loop - subjects)
        end % (p loop - discovered paths)
    end % (j loop - folders)

    % --- merge for this file type finished ---

    % remove the temporary counter fields
    allMergedFields = fieldnames(mergedData);
    counterFields = allMergedFields(startsWith(allMergedFields, 'counter_'));
    if ~isempty(counterFields)
        mergedData = rmfield(mergedData, counterFields);
    end

    % save under a new file name
    outputFileName = ['merged_' currentMatFile];

    % create a variable with the original struct name
    eval(sprintf('%s = mergedData;', currentStructName));

    % save only that variable to the file
    try
        fprintf('\n  Saving merged file: %s ...\n', outputFileName);
        % save(outputFileName, currentStructName, '-v7.3');
        fprintf('  Saved!\n\n');
    catch ME
        fprintf('  File save error: %s. Error: %s\n\n', outputFileName, ME.message);
    end

    % cleanup is done once after the loop, so no clear here
    % clear(currentStructName);

end % (i loop - file types)

fprintf('--------------------------------------------------\n');
fprintf('All data merges finished.\n');
fprintf('Generated files:\n');
for i = 1:length(matFileNames)
    fprintf('- merged_%s\n', matFileNames{i});
end

% --- completion and variable cleanup ---
%
% Remove every variable used by this script from the workspace except the
% final structs listed in targetStructNames
% (e.g. 'stride_modeling', 'stride_grf', ...).

fprintf('\nCleaning up the temporary variables used by this script...\n');

% join targetStructNames into one space-separated string
keepVarsStr = strjoin(targetStructNames, ' ');

% dynamically build the 'clearvars -except [var1] [var2] ...' command
clearCmd = ['clearvars -except ', keepVarsStr];

% execute the clear command via eval
try
    eval(clearCmd);
    fprintf('Variable cleanup complete. (only the 4 merged structs remain)\n');
catch ME_clear
    warning('Variable cleanup failed: %s\n', ME_clear.message);
end

%% Remove the S011 / ss3 data
fprintf('\nRemoving Subject S011, Session ss3 from every struct...\n');

subID_to_remove = 'S011';
sess_to_remove  = 'ss3';

% the four struct names left in the workspace after the merge
structNames = {
    'stride_modeling', ...
    'stride_kinematics_arm_vector', ...
    'stride_kinematics_arm', ...
    'stride_grf' ...
};

for idxVar = 1:numel(structNames)
    varName = structNames{idxVar};

    % check that the variable actually exists
    if ~exist(varName, 'var')
        continue;
    end

    Sdata = eval(varName);
    if ~isstruct(Sdata)
        continue;
    end

    topFields = fieldnames(Sdata);

    % --------- step 1: Sxxx at the top level (stride_grf, stride_modeling layouts) ---------
    hasTopSubject = false;
    for ii = 1:numel(topFields)
        if strncmp(topFields{ii}, 'S', 1)
            hasTopSubject = true;
            break;
        end
    end

    if hasTopSubject
        % subject level at the top: remove S011.ss3
        if isfield(Sdata, subID_to_remove)
            subjBlock = Sdata.(subID_to_remove);
            if isstruct(subjBlock) && isfield(subjBlock, sess_to_remove)
                subjBlock = rmfield(subjBlock, sess_to_remove);
                Sdata.(subID_to_remove) = subjBlock;
                fprintf('  - %s: removed %s.%s (top-level)\n', ...
                    varName, subID_to_remove, sess_to_remove);
            end
        end

    else
        % --------- step 2: only segments at the top level (e.g. L_Wrist, L_Shoulder_to_sacrum) ---------
        %   → remove S011.ss3 under each segment (stride_kinematics_arm, stride_kinematics_arm_vector layouts)

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
                    fprintf('  - %s: removed %s.%s.%s\n', ...
                        varName, segName, subID_to_remove, sess_to_remove);
                end
            end
        end
    end

    % write the changes back to the original variable
    eval([varName ' = Sdata;']);
end

fprintf('S011 / ss3 removal finished.\n');

try
    save('merged_stride_modeling.mat',                'stride_modeling',           '-v7.3');
    save('merged_stride_kinematics_arm_vector_gyro.mat','stride_kinematics_arm_vector','-v7.3');
    save('merged_stride_kinematics_arm_gyro_offset_added.mat','stride_kinematics_arm','-v7.3');
    save('merged_stride_grf_gyro.mat',                 'stride_grf',               '-v7.3');
    fprintf('merged_*.mat files re-saved after removing S011 / ss3.\n');
catch ME_save
    warning('Error while saving after the S011 / ss3 removal: %s', ME_save.message);
end

% -------------------------------------------------------------------
% --- local function definitions (must sit at the bottom of the script) ---
% -------------------------------------------------------------------

function paths = findSubjectPaths(currentStruct)
    % Recursively search the struct (currentStruct) and return every sub-path
    % that contains 'S...' fields as a cell array (paths).
    % e.g. {'L_Wrist', 'R_Wrist'} or {'L_Wrist.Foo', 'R_Wrist.Bar'},
    % or {''} when the subjects sit at the top level.

    paths = {};

    try
        fields = fieldnames(currentStruct);
    catch
        % stop if currentStruct is not a struct (e.g. a numeric array)
        return;
    end

    if isempty(fields)
        return;
    end

    % --- criterion 1: is the current level the 'S...' subject level? ---
    % (heuristic: treat it as the subject level when >50% of fields start with 'S')
    sFields = startsWith(fields, 'S');
    if any(sFields)
        if (sum(sFields) / length(fields)) > 0.5
            paths = {''}; % found: the current path is the leaf (return an empty string)
            return;
        end
    end

    % --- criterion 2: otherwise, recurse into the sub-structs ---
    for i = 1:length(fields)
        fieldName = fields{i};

        % fields starting with 'S' are not recursion targets (layouts like S001.data)
        if startsWith(fieldName, 'S')
            continue;
        end

        % check that the child field is a struct
        if isstruct(currentStruct.(fieldName))
            % recursive call
            subPaths = findSubjectPaths(currentStruct.(fieldName));

            % combine the returned paths with the current path
            for j = 1:length(subPaths)
                if isempty(subPaths{j})
                    % the child returned {''} (e.g. found 'L_Wrist')
                    paths{end+1} = fieldName;
                else
                    % the child returned {'Foo'} etc. (e.g. found 'L_Wrist.Foo')
                    paths{end+1} = [fieldName '.' subPaths{j}];
                end
            end
        end
    end

    % remove duplicate paths (defensive)
    paths = unique(paths, 'stable');
end
