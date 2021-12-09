%%% Image Segmentation
%%% Christian Morrell, Alexandre Banks
%%% November 11, 2021

%%% Goal: We want to take a blood slide image and produce an individual
%%% image for each cell within the full field image

close all
clear
clc

%% Load in full field images

% Load raw data
numImages = 50; % Number of images in database 2
fullFiles = cell(numImages, 1); % Preallocate array
fullImages = cell(numImages, 1);    % Preallocate array
imageName = '\source.jpg';   % Name of source images in dataset

for i = 1:numImages
    
    if i < 10
        % Need to append 0 to find proper file folder
        dirName = ['erythrocytesIDB2\0' num2str(i) 'erythrocytesIDB2'];
    else
        dirName = ['erythrocytesIDB2\' num2str(i) 'erythrocytesIDB2'];
    end
    
    fullFiles{i} = dir([dirName imageName]);    % Get jpg file
    temp  = readimagefiles(fullFiles{i}, dirName);  % Load image
    fullImages{i} = temp{1}; 
    
end

% Convert RGB images to grayscale
grayImages = ConvRGB_to_GRAY(fullImages)';

% Split up full field images into datasets
segmented = segmentFullFieldImages(grayImages);




