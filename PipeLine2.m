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
LDAImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
DTImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
QDAImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
NBImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
SVMImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
kNNImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images


for i = 1:numImages
    
    currentFeatureSet = featureSet{i};  % Get current feature set
    numCells = height(currentFeatureSet);
    LDAResults = zeros(numCells, 1);  % Store results of each cell in image
    DTResults = zeros(numCells, 1);  % Store results of each cell in image
    QDAResults = zeros(numCells, 1);  % Store results of each cell in image
    NBResults = zeros(numCells, 1);  % Store results of each cell in image
    SVMResults = zeros(numCells, 1);  % Store results of each cell in image
    kNNResults = zeros(numCells, 1);  % Store results of each cell in image
    
    
    for j = 1:numCells
        %         LDAResults(j) = predict(LDA_B_Model, currentFeatureSet(j, :));  % Classify new data
        LDAResults(j) = predict(LDA_B_Model, currentFeatureSet(j, :));  % Classify new data
        DTResults(j) = predict(DT_B_Model, currentFeatureSet(j, :));  % Classify new data
        QDAResults(j) = predict(QDA_B_Model, currentFeatureSet(j, :));  % Classify new data
        NBResults(j) = predict(NB_B_Model, currentFeatureSet(j, :));  % Classify new data
        SVMResults(j) = predict(SVM_B_Model, currentFeatureSet(j, :));  % Classify new data
        kNNResults(j) = predict(kNN_B_Model, currentFeatureSet(j, :));  % Classify new data
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

classifierResults = [LDAImageClassifications DTImageClassifications QDAImageClassifications ...
    NBImageClassifications SVMImageClassifications kNNImageClassifications];


