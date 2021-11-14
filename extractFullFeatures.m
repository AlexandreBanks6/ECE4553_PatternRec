function [featureSet, featureNames] = extractFullFeatures(segmentedDataset)
numImages = length(segmentedDataset);
featureSet = cell(numImages, 1);    % Hold features for each full field image
featureNames = cell(numImages, 1);    % Hold features for each full field image


for i = 1:numImages
    cellImages = segmentedDataset{i};  % Get cell images for one full field image
    % Extract features
    [featureSet{i}, featureNames{i}] = extractCellFeatures(cellImages');
end

save('FeatureSet.mat', 'featureSet', 'featureNames')    % Save variables
end

