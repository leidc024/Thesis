% ============================================================
% disambiguate_candidates.m
% Use Python disambiguation model instead of defaulting to first candidate
% ============================================================
%
% USAGE (inside MaBaybay after getting transliterations):
%
%   % Instead of using first candidate:
%   % result = transliterations{1};
%   
%   % Use disambiguation:
%   result = disambiguate_candidates(transliterations);
%
% ============================================================

function result = disambiguate_candidates(transliterations)
    % transliterations: cell array where each cell contains candidate(s)
    %   e.g., {{'dito','rito'}, {'ang'}, {'lugar','logar'}}
    %
    % Returns: cell array of disambiguated words
    %   e.g., {'dito', 'ang', 'lugar'}
    
    % Path to Python script and virtual environment
    thesis_dir = fullfile(fileparts(mfilename('fullpath')), '..', '..');
    python_script = fullfile(thesis_dir, 'disambiguate.py');
    python_exe = fullfile(thesis_dir, '.venv', 'Scripts', 'python.exe');  % Windows venv
    temp_json = fullfile(tempdir, 'mabaybay_candidates.json');
    
    % Convert transliterations to JSON format
    candidates = cell(1, length(transliterations));
    for i = 1:length(transliterations)
        word_cands = transliterations{i};
        if ischar(word_cands)
            candidates{i} = {word_cands};
        elseif iscell(word_cands)
            candidates{i} = word_cands(:)';
        else
            candidates{i} = {char(word_cands)};
        end
    end
    
    % Save to temp JSON
    json_str = jsonencode(candidates);
    fid = fopen(temp_json, 'w');
    fprintf(fid, '%s', json_str);
    fclose(fid);
    
    % Call Python disambiguator (using virtual environment)
    % NOTE: First call takes ~15-30 seconds to load RoBERTa model
    fprintf('Running context-aware disambiguation (may take 15-30 sec on first run)...\n');
    fprintf('JSON saved to: %s\n', temp_json);
    fprintf('Candidates JSON: %s\n', json_str);
    python_cmd = sprintf('"%s" "%s" "%s"', python_exe, python_script, temp_json);
    [status, output] = system(python_cmd);
    
    fprintf('Python status: %d\n', status);
    fprintf('Python output: [%s]\n', output);
    
    if status ~= 0
        warning('Disambiguation failed, using first candidates. Error: %s', output);
        % Fallback to first candidate
        result = cell(1, length(transliterations));
        for i = 1:length(transliterations)
            word_cands = transliterations{i};
            if iscell(word_cands)
                result{i} = word_cands{1};
            else
                result{i} = char(word_cands);
            end
        end
        return;
    end
    
    % Parse output (space-separated words)
    output = strtrim(output);
    result = strsplit(output, ' ');
    
    % Cleanup
    delete(temp_json);
end
