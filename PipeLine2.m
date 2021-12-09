%%% PipeLine2.m
%%% Christian Morrell, Alexandre Banks
%%% ECE4553 Project
%%% Integrate image segmentation and Pipeline1.m

close all
clear
clc

%% Load data

load Classifiers_new.mat    % Get trained classifiers
load NeuralNetResults.mat


% Database 3
DB2Images = 50;
DB3Images = 30;

DBName = 'erythrocytesIDB2';    % choose which database to read in

if strcmp(DBName, 'erythrocytesIDB2')
    numImages = DB2Images;
    grayImages = loadFullFieldData(DBName, DB2Images);
elseif strcmp(DBName, 'erythrocytesIDB3')
    grayImages = loadFullFieldData(DBName, DB3Images);
    
    % Get images from BCCD dataset
    imageFiles = dir('TestDataset\Images\*.jpg');
    nFiles = length(imageFiles);
    normalImages = 300; % number of images in BCCD dataset
    fullImages = cell(normalImages, 1);    % Preallocate array
    for i = 1:nFiles
        name = imageFiles(i).name;
        image = imread(['TestDataset\Images\' name]);
        fullImages{i} = image;
    end
    
    newGray = ConvRGB_to_GRAY(fullImages');
    
    % Add BCCD images to sickle images for full test dataset
    for i = 1:normalImages
        grayImages{i + DB3Images} = newGray{i};
    end
    
    numImages = normalImages + DB3Images;   % total number of images in dataset
end


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

rankedFeats = cell(size(featureSet));   % features ranked based off of mRMR algorithm
mRMRFeats = cell(size(featureSet)); % Features after feature selection

for i = 1:numImages
    rankedFeats{i} = featureSet{i}(:, featureIdx);
    mRMRFeats{i} = rankedFeats{i}(:, 1:numFeats);
end


%% Project Features Onto ULDA Feature Space

ULDAmRMRFeatures = cell(numImages, 1); % Array to hold ULDA feature space


for i = 1:numImages
    ULDAmRMRFeatures{i} = mRMRFeats{i} * ULDA_Weight;
end


%% Use classifier to determine if blood sample has sickle cell anemia

%
thresholds = [0.05 0.1 0.15 0.2 0.25 0.3 0.35];
ROCX = cell(size(thresholds));
ROCY = cell(size(thresholds));

if strcmp(DBName, 'erythrocytesIDB2')
    labels = 2 * ones(numImages, 1);    % every image is sickle
elseif strcmp(DBName, 'erythrocytesIDB3')
    labels = ones(numImages, 1);
    normalInDB3 = [11, 12, 14, 15, 16, 17]; % images that were observed to be normal in DB3
    labels(1:DB3Images) = 2;    % sickle
    labels(normalInDB3) = 1;
end

for threshLoopCounter = 1:length(thresholds)
    
    threshold = thresholds(threshLoopCounter);
    % Pass it into classifier
    LDAImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
    DTImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
    QDAImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
    NBImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
    SVMImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
    SVMImageScores = zeros(numImages, 2);
    kNNImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
    CNNImageClassifications = zeros(numImages, 1);    % Array to hold final classifications of full images
    
    for i = 1:numImages
        
        currentFeatureSet = ULDAmRMRFeatures{i};  % Get current feature set
        numCells = height(currentFeatureSet);
        LDAResults = zeros(numCells, 1);  % Store results of each cell in image
        DTResults = zeros(numCells, 1);  % Store results of each cell in image
        QDAResults = zeros(numCells, 1);  % Store results of each cell in image
        NBResults = zeros(numCells, 1);  % Store results of each cell in image
        SVMResults = zeros(numCells, 1);  % Store results of each cell in image
        SVMScores = zeros(numCells, 2);
        kNNResults = zeros(numCells, 1);  % Store results of each cell in image
        CNNResults = zeros(numCells, 1);  % Store results of each cell in image
        
        for j = 1:numCells
            LDAResults(j) = predict(LDA_ULDA_mRMR_Model, currentFeatureSet(j, :));  % Classify new data
            DTResults(j) = predict(DT_ULDA_mRMR_Model, currentFeatureSet(j, :));  % Classify new data
            QDAResults(j) = predict(QDA_ULDA_mRMR_Model, currentFeatureSet(j, :));  % Classify new data
            NBResults(j) = predict(NB_ULDA_mRMR_Model, currentFeatureSet(j, :));  % Classify new data
            [SVMResults(j), SVMScores(j, :)]  = predict(SVM_ULDA_mRMR_Model, currentFeatureSet(j, :));  % Classify new data
            kNNResults(j) = predict(kNN_ULDA_mRMR_Model, currentFeatureSet(j, :));  % Classify new data
            CNNResults(j) = classify(net, imresize(segmentedDataset{i}{j}, [80 80]));  % Classify new data
        end
        
        % Get number of sickle cells
        LDANumSickle = length(find(LDAResults == 2));
        DTNumSickle = length(find(DTResults == 2));
        QDANumSickle = length(find(QDAResults == 2));
        NBNumSickle = length(find(NBResults == 2));
        SVMNumSickle = length(find(SVMResults == 2));
        kNNNumSickle = length(find(kNNResults == 2));
        CNNumSickle = length(find(CNNResults == 2));
        
        % Compare ratio of normal to sickle to threshold
        LDARatio = LDANumSickle/numCells;
        DTRatio = DTNumSickle/numCells;
        QDARatio = QDANumSickle/numCells;
        NBRatio = NBNumSickle/numCells;
        SVMRatio = SVMNumSickle/numCells;
        kNNRatio = kNNNumSickle/numCells;
        CNNRatio = CNNumSickle/numCells;
        
        % Store image classifications
        % 1 = normal
        % 2 = sickle
        % 3 = rejection
        
        LDAImageClassifications(i) = fullFieldResult(LDARatio, threshold);
        DTImageClassifications(i) = fullFieldResult(DTRatio, threshold);
        QDAImageClassifications(i) = fullFieldResult(QDARatio, threshold);
        NBImageClassifications(i) = fullFieldResult(NBRatio, threshold);
        SVMImageClassifications(i) = fullFieldResult(SVMRatio, threshold);
        kNNImageClassifications(i) = fullFieldResult(kNNRatio, threshold);
        CNNImageClassifications(i) = fullFieldResult(CNNRatio, threshold);
        SVMImageScores(i, :) = mean(SVMScores);
    end
    
    % ROC data for SVM
    classifiedIdx = find(SVMImageClassifications == 1 | SVMImageClassifications == 2);  % remove rejections
    ROCScores = SVMImageScores(classifiedIdx, :);
    [ROCX{threshLoopCounter}, ROCY{threshLoopCounter}] = perfcurve(labels(classifiedIdx), ROCScores(:, 2), 2);
    
    
end



%% Display results

% Summarize results
classifierResults = [LDAImageClassifications DTImageClassifications QDAImageClassifications ...
    NBImageClassifications SVMImageClassifications kNNImageClassifications];

% Store as strings for readability
resultsStrings = strings(size(classifierResults));  % Strings array to hold results
resultsStrings(:, :) = "Normal";    % Set all elements to normal
sickleIdx = find(classifierResults == 2);    % Find which elements are sickle
rejectIdx = find(classifierResults == 3);   % Find which elements were rejected
resultsStrings(sickleIdx) = "Sickle";   % Set corresponding elements to sickle
resultsStrings(rejectIdx) = "Rejected"; % Set corresponding elements to rejected

% Classifier accuracy
nbAcc = fullFieldAccuracy(NBImageClassifications(NBImageClassifications == 1 | NBImageClassifications == 2), ...
    labels(NBImageClassifications == 1 | NBImageClassifications == 2));
ldaAcc = fullFieldAccuracy(LDAImageClassifications(LDAImageClassifications == 1 | LDAImageClassifications == 2), ...
    labels(LDAImageClassifications == 1 | LDAImageClassifications == 2));
dtAcc = fullFieldAccuracy(DTImageClassifications(DTImageClassifications == 1 | DTImageClassifications == 2), ...
    labels(DTImageClassifications == 1 | DTImageClassifications == 2));
qdaAcc = fullFieldAccuracy(QDAImageClassifications(QDAImageClassifications == 1 | QDAImageClassifications == 2), ...
    labels(QDAImageClassifications == 1 | QDAImageClassifications == 2));
svmAcc = fullFieldAccuracy(SVMImageClassifications(SVMImageClassifications == 1 | SVMImageClassifications == 2), ...
    labels(SVMImageClassifications == 1 | SVMImageClassifications == 2));
knnAcc = fullFieldAccuracy(kNNImageClassifications(kNNImageClassifications == 1 | kNNImageClassifications == 2), ...
    labels(kNNImageClassifications == 1 | kNNImageClassifications == 2));
cnnAcc = fullFieldAccuracy(CNNImageClassifications(CNNImageClassifications == 1 | CNNImageClassifications == 2), ...
    labels(CNNImageClassifications == 1 | CNNImageClassifications == 2));

classifierFullFieldAccuracies = [nbAcc ldaAcc dtAcc qdaAcc svmAcc knnAcc cnnAcc]';

% Bar graph of classifier accuracies for full-field images
classifierNames = {'NB', 'LDA', 'DT', 'QDA', 'SVM', 'kNN', 'CNN'};
barLabels=categorical(classifierNames);
barLabels=reordercats(barLabels, classifierNames);  % reorder categorical array

figure;
bar(barLabels, classifierFullFieldAccuracies);   %Plotting bar chat of features with associated percentages
title('Comparison of Classification Methods Applied to Full-field Images')
ylabel('Classification Accuracy')

% ROC for threshold
figure

for i = 1:length(thresholds)
    if i == length(thresholds)
        plot(ROCX{i}, ROCY{i}, 'LineWidth', 2)  % highlight best curve
    end
    plot(ROCX{i}, ROCY{i}, '--')
    hold on
    grid on
end
lgd = legend('0.05', '0.1', '0.15', '0.2', '0.25', '0.3', '0.35', ...
    'Location', 'southeast');
title(lgd, 'Threshold Values')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for SVM Classifier with Varying Threshold')

% Confusion matrix
figure
C = confusionmat(labels, SVMImageClassifications);
confusionchart(C)
title('Confusion Matrix for SVM Classifier with Threshold of 0.35')
figure
C = confusionmat(labels, CNNImageClassifications);
confusionchart(C)
title('Confusion Matrix for CNN with Threshold of 0.35')



