%%% PipeLine2.m
%%% Christian Morrell, Alexandre Banks
%%% ECE4553 Project
%%% Integrate image segmentation and Pipeline1.m

close all
clear
clc

%% Load data

load Classifiers_new.mat    % Get trained classifiers

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

%% Feature ranking

% rankedFeats = featureSet(:, featureIdx);  % features ranked based off of mRMR algorithm

rankedFeats = cell(size(featureSet));   % features ranked based off of mRMR algorithm
mRMRFeats = cell(size(featureSet)); % Features after feature selection

for i = 1:numImages
    rankedFeats{i} = featureSet{i}(:, featureIdx);
    mRMRFeats{i} = rankedFeats{i}(:, 1:numFeats);
end

% mRMRFeats = Features_mrmr(:, 1:numFeats); % Features after feature selection

%% Project Features Onto ULDA Feature Space

ULDAFeatures = cell(numImages, 1); % Array to hold ULDA feature space

for i = 1:numImages
    ULDAFeatures{i} = featureSet{i} * ULDA_W;
end


%% Use classifier to determine if blood sample has sickle cell anemia

threshold = 0.05;   % Threshold to determine if image should be flagged for sickle cell

% Pass it into classifier
LDAImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
DTImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
QDAImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
NBImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
SVMImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
kNNImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images


for i = 1:numImages
    
    currentFeatureSet = ULDAFeatures{i};  % Get current feature set
    numCells = height(currentFeatureSet);
    LDAResults = zeros(numCells, 1);  % Store results of each cell in image
    DTResults = zeros(numCells, 1);  % Store results of each cell in image
    QDAResults = zeros(numCells, 1);  % Store results of each cell in image
    NBResults = zeros(numCells, 1);  % Store results of each cell in image
    SVMResults = zeros(numCells, 1);  % Store results of each cell in image
    kNNResults = zeros(numCells, 1);  % Store results of each cell in image
    
    
    for j = 1:numCells
        LDAResults(j) = predict(LDA_ULDA_mRMR_Model, currentFeatureSet(j, :));  % Classify new data
        DTResults(j) = predict(DT_ULDA_mRMR_Model, currentFeatureSet(j, :));  % Classify new data
        QDAResults(j) = predict(QDA_ULDA_mRMR_Model, currentFeatureSet(j, :));  % Classify new data
        NBResults(j) = predict(NB_ULDA_mRMR_Model, currentFeatureSet(j, :));  % Classify new data
        SVMResults(j) = predict(SVM_ULDA_mRMR_Model, currentFeatureSet(j, :));  % Classify new data
        kNNResults(j) = predict(kNN_ULDA_mRMR_Model, currentFeatureSet(j, :));  % Classify new data
    end
    
    % Get number of sickle cells
    LDANumSickle = length(find(LDAResults == 2));
    DTNumSickle = length(find(DTResults == 2));
    QDANumSickle = length(find(QDAResults == 2));
    NBNumSickle = length(find(NBResults == 2));
    SVMNumSickle = length(find(SVMResults == 2));
    kNNNumSickle = length(find(kNNResults == 2));
    
    % Compare ratio of normal to sickle to threshold
    LDARatio = LDANumSickle/numCells;
    DTRatio = DTNumSickle/numCells;
    QDARatio = QDANumSickle/numCells;
    NBRatio = NBNumSickle/numCells;
    SVMRatio = SVMNumSickle/numCells;
    kNNRatio = kNNNumSickle/numCells;
    
    % Store image classifications
    LDAImageClassifications(i) = LDARatio > threshold;
    DTImageClassifications(i) = DTRatio > threshold;
    QDAImageClassifications(i) = QDARatio > threshold;
    NBImageClassifications(i) = NBRatio > threshold;
    SVMImageClassifications(i) = SVMRatio > threshold;
    kNNImageClassifications(i) = kNNRatio > threshold;
end

% Summarize results
classifierResults = [LDAImageClassifications DTImageClassifications QDAImageClassifications ...
    NBImageClassifications SVMImageClassifications kNNImageClassifications];

% Store as strings for readability
resultsStrings = strings(size(classifierResults));  % Strings array to hold results
resultsStrings(:, :) = "Normal";    % Set all elements to normal
sickleIdx = find(classifierResults);    % Find which elements are sickle
resultsStrings(sickleIdx) = "Sickle";   % Set corresponding elements to sickle

% Display classifications as histogram
figure()
barLabels = categorical({'LDA', 'DT', 'QDA', 'NB', 'SVM', 'kNN'});  % Labels for Bar Graph
bar(barLabels, sum(classifierResults))
xlabel('Classifier')
ylabel('Number of Images Diagnosed with Sickle Cell')
title('Number of Images Diagnosed with Sickle Cell for Each Classifier')


