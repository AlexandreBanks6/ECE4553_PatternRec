%%% CNN.m
%%% ECE4553 Project
%%% Training Convolutional Neural Network
%%% Alexandre Banks, Christian Morrell
%%% November 27 2021

close all
clear
clc

%% Load and Explore Image Data

% Set path
cellDatasetPath = fullfile('Version 2, erythrocytesIDB 2021', 'Version 2, erythrocytesIDB 2021', 'erythrocytesIDB1', 'individual cells');

% Load images
cellDatastore = imageDatastore(cellDatasetPath, ... 
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Calculate the number of images in each class
labelCount = countEachLabel(cellDatastore);



%% Specify Training and Validation Sets

trainPerc = 0.7;    % percentage of data to be training data
% testPerc = 1 - trainPerc;   % percentage of data to be testing data


[imgsTrain, imgsValidation] = splitEachLabel(cellDatastore, trainPerc, 'randomize');



%% Define Network Architecture

% CNN parameters
filterSize = 3; % size of filter for convolutional layer
numFilters = 80; % number of filters for convolutional layer
poolSize = numFilters;   % pool size for pooling layer
strideNum = 2;  % stride value for pooling layer
numClasses = 2; % number of classes to output

% Define layers of CNN based on LeNet
% layers = [
%     imageInputLayer(size(readimage(cellDatastore, 1)))
%     
%     convolution2dLayer(filterSize, numFilters, 'Padding', 'same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(poolSize/2, 'Stride', strideNum);
%     
%     convolution2dLayer(filterSize, numFilters/4, 'Padding', 'same')
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(poolSize/8, 'Stride', strideNum);
%     
%     fullyConnectedLayer(5)
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer
% ];

layers = [
    imageInputLayer([80 80 3])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];


%% Specify Training Options

% options = trainingOptions('sgdm', ...
%     'InitialLearnRate',0.01, ...
%     'MaxEpochs',20, ...
%     'MiniBatchSize', 256, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData',imgsValidation, ...
%     'ValidationFrequency',30, ...
%     'Verbose',false, ...
%     'Plots','training-progress');

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',16, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imgsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Pre-process Images

imageSize = size(readimage(cellDatastore, 1));
augmentedTrainingSet = augmentedImageDatastore(imageSize, imgsTrain, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, imgsValidation, 'ColorPreprocessing', 'gray2rgb');

%% Train CNN
net = trainNetwork(augmentedTrainingSet, layers, options);

%% Accuracy
YPred = classify(net, augmentedTestSet);
YTest = imgsValidation.Labels;

accuracy = sum(YPred == YTest) / numel(YTest);




%% Extract Training Features

% featureLayer = 'fc8';
% trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, ...
%     'MiniBatchSize', 32, 'OutputAs', 'columns');
% numFeatures = 439;









