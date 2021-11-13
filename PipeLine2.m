%%% PipeLine2.m
%%% Christian Morrell, Alexandre Banks
%%% ECE4553 Project
%%% Integrate image segmentation and pipeline 1

close all
clear
clc

%% Load data

load Classifiers.mat    % Get trained classifiers

% Database 2 (full field images)
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


%% Segment full field images into individual cells

segmented = segmentFullFieldImages(grayImages);

%% Feature extraction

% Get each individual cell from image
cellImages = segmented{1};  % Get cell images for one full field image

% Extract features
[featureSet, featureNames] = extractCellFeatures(cellImages');

%%% FEATURE SELECTION???????????????????????????????????????

%% Using ULDA For Dimensionality Reduction
PercGoal=95;    % 95 percent of total variance explained by projected data

[ULDA_Features,explained,ProjDatUnCleaned]=ULDA(featureSet,labels,PercGoal);

%% Use classifier to determine if blood sample has sickle cell anemia


% Pass it into classifier
% exampleImage = cellImages{1};   % Get cell image
% result = predict(ULDA_Features, exampleImage);



% Tally result
% results(i) = result;  % (1 = normal, 2 = sickle)
% Repeat until all cells in an image are done

% Compare ratio of normal to sickle to threshold
% numSickle = length(find(results == 1));
% ratio = numSickle/numCells;
% if ratio > threshold THEN image is sickle




