%%% PipeLine2.m
%%% Christian Morrell, Alexandre Banks
%%% ECE4553 Project
%%% Integrate image segmentation and pipeline 1

close all
clear
clc

%% Script parameters
segmentImages = false;  % logical to determine if images need to be segmented again
extractFeatures = false;    % logical to determine if features need to be extracted

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
if isfile('Segmented.mat')
    % File exists
    load Segmented.mat
else
    % Dataset needs to be segmented
    segmentedDataset = segmentFullFieldImages(grayImages); 
end


%% Feature extraction
if isfile('FeatureSet.mat')
    % File exists
    load FeatureSet.mat
else
    % Features need to be extracted
    [featureSet, featureNames] = extractFullFeatures(segmentedDataset);
end



%% Use classifier to determine if blood sample has sickle cell anemia

threshold = 0.05;   % Threshold to determine if image should be flagged for sickle cell

% Pass it into classifier
imageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images


for i = 1:numImages

currentFeatureSet = featureSet{i};  % Get current feature set
numCells = height(currentFeatureSet);
results = zeros(numCells, 1);  % Store results of each cell in image
for j = 1:numCells
    results(j) = predict(LDA_Model, currentFeatureSet(j, :));
end
    % Compare ratio of normal to sickle to threshold
    numSickle = length(find(results == 2));
    ratio = numSickle/numCells;
    imageClassifications(i) = ratio > threshold;
end




