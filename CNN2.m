clear
clc


%% Read In Grayscale Images

% Set path
cellDatasetPath = fullfile('Version 2, erythrocytesIDB 2021', 'Version 2, erythrocytesIDB 2021', 'erythrocytesIDB1', 'individual cells');

% Load images
cellDatastore = imageDatastore(cellDatasetPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%% Split the dataset
[CellTrain,CellValidate]=splitEachLabel(cellDatastore,0.66,'randomize');


%% Setting layers of CNN
%Setting Parameters
inputSize=[80,80,1];    %Input image is an 80x80 grayscale
FilterSize=3;
ConvolutionStride=1;
InitialNumFilter=32;
pool_size=4;
stride_size=2;
classSize=2;

layers=[
    

%-------------------<Classifer Resembling LetNet>----------------------

imageInputLayer(inputSize); %Defines the input size

%First 2 convolutional layers
convolution2dLayer(FilterSize,InitialNumFilter,'Padding','same','Stride',ConvolutionStride);   %Input size is the same as output size
batchNormalizationLayer
reluLayer

convolution2dLayer(FilterSize,InitialNumFilter/2,'Padding','same','Stride',ConvolutionStride);   %Input size is the same as output size
batchNormalizationLayer
reluLayer

maxPooling2dLayer(pool_size,'Stride',stride_size);


%Second 2 convolutional layers
convolution2dLayer(FilterSize,InitialNumFilter/4,'Padding','same','Stride',ConvolutionStride);   %Input size is the same as output size
batchNormalizationLayer
reluLayer

convolution2dLayer(FilterSize,InitialNumFilter/8,'Padding','same','Stride',ConvolutionStride);   %Input size is the same as output size
batchNormalizationLayer
reluLayer

maxPooling2dLayer(pool_size,'Stride',stride_size);



fullyConnectedLayer(classSize)
fullyConnectedLayer(classSize)
softmaxLayer
classificationLayer


];

augmentedTrainingSet = augmentedImageDatastore(inputSize, CellTrain, 'ColorPreprocessing', 'rgb2gray');
augmentedTestSet = augmentedImageDatastore(inputSize, CellValidate, 'ColorPreprocessing', 'rgb2gray');


%% Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',16, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augmentedTrainingSet, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');


%% Training Network
net=trainNetwork(augmentedTrainingSet,layers,options);

%% Save The Trained Network
save('NeuralNetResults.mat','net','augmentedTestSet');





