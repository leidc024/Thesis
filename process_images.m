% =========================================================================
% SCRIPT: process_images.m (Version 2)
% PURPOSE: This script automates the process of running MaBaybay-OCR on a
%          folder of Baybayin images and saves the structured output
%          (including all ambiguous candidates) to a single JSON file.
% =========================================================================

function process_images()
    % --- CONFIGURATION ---
    inputImageFolder = 'baybayin_dataset_images'; % Folder with your PNGs
    outputJsonFile = 'ocr_results.json';         % The final JSON output file
    maBaybayRootFolder = 'MaBaybay-OCR';         % Path to the main OCR package folder

    % --- 1. SETUP THE ENVIRONMENT ---
    % Add the entire MaBaybay-OCR folder and all its subfolders to the path.
    % This is a more robust way to ensure all functions and files are found.
    if ~isfolder(maBaybayRootFolder)
        error('MaBaybay-OCR folder not found. Please ensure it is in the same directory as this script.');
        return;
    end
    addpath(genpath(maBaybayRootFolder));
    fprintf('Added MaBaybay-OCR package to the path.\n');

    % --- 2. FIND ALL IMAGES TO PROCESS ---
    imageFilesPNG = dir(fullfile(inputImageFolder, '*.png'));
    imageFilesJPG = dir(fullfile(inputImageFolder, '*.jpg'));
    allImageFiles = [imageFilesPNG; imageFilesJPG];

    if isempty(allImageFiles)
        error('No image files (.png or.jpg) found in the folder: %s', inputImageFolder);
        return;
    end
    
    fprintf('Found %d images to process.\n', length(allImageFiles));

    % --- 3. PROCESS EACH IMAGE AND COLLECT RESULTS ---
    allResults = {};

    for i = 1:length(allImageFiles)
        currentFilename = allImageFiles(i).name;
        fullImagePath = fullfile(inputImageFolder, currentFilename);
        
        fprintf('Processing image %d of %d: %s\n', i, length(allImageFiles), currentFilename);

        try
            % Call the core MaBaybay-OCR function. The '1' argument is an
            % option that typically enables the more robust processing.
            transliteratedTokens = Baybayin_word_reader(fullImagePath, 1);
            
            % Create a struct to hold the results for this image.
            imageResult.image_filename = currentFilename;
            imageResult.tokens = transliteratedTokens;
            
            % Add the result to our main collection.
            allResults{end+1} = imageResult;

        catch ME
            % Provide a more detailed error message if something goes wrong.
            warning('Could not process image: %s.\nError message: %s\nCheck if all classifier (.mat) files are in the MaBaybay-OCR/Algorithms/ folder.', currentFilename, ME.message);
            continue; % Skip to the next image
        end
    end

    % --- 4. CONVERT RESULTS TO JSON AND SAVE TO FILE ---
    fprintf('\nOCR processing complete. Converting results to JSON...\n');
    
    % Use jsonencode to convert the cell array of structs into a JSON string.
    jsonString = jsonencode(allResults, 'PrettyPrint', true);

    % Write the JSON string to the output file.
    try
        fileID = fopen(outputJsonFile, 'w');
        fprintf(fileID, '%s', jsonString);
        fclose(fileID);
        fprintf('Successfully saved structured data to: %s\n', outputJsonFile);
    catch ME
        error('Failed to write JSON file. Error: %s', ME.message);
    end
end