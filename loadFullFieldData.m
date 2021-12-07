function grayImages = loadFullFieldData(DBName, numImages)
fullFiles = cell(numImages, 1); % Preallocate array
fullImages = cell(numImages, 1);    % Preallocate array
imageName = '\source.jpg';   % Name of source images in dataset

for i = 1:numImages
    
    if i < 10
        % Need to append 0 to find proper file folder
        dirName = ['TestDataset\' DBName '\0' num2str(i) DBName];
    else
        dirName = ['TestDataset\' DBName '\' num2str(i) DBName];
    end
    
    fullFiles{i} = dir([dirName imageName]);    % Get jpg file
    temp  = readimagefiles(fullFiles{i}, dirName);  % Load image
    fullImages{i} = temp{1};
    
end

% Convert RGB images to grayscale
grayImages = ConvRGB_to_GRAY(fullImages)';
end

