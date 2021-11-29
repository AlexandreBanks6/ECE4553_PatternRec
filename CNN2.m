clear
clc


%% Read In Grayscale Images

%ImageDatasetPath=fullfile('GrayImages'); %Folder containing subfolders for different classes

%ImageData=imageDatastore(ImageDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');

% Set path
cellDatasetPath = fullfile('Version 2, erythrocytesIDB 2021', 'Version 2, erythrocytesIDB 2021', 'erythrocytesIDB1', 'individual cells');

% Load images
cellDatastore = imageDatastore(cellDatasetPath, ... 
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%% Split the dataset
[CellTrain,CellValidate]=splitEachLabel(cellDatastore,0.7,'randomize');


%% Setting layers of CNN
%Setting Parameters
inputSize=[80,80,3];    %Input image is an 80x80 grayscale
FilterSize=3;
ConvolutionStride=1;
InitialNumFilter=32;
pool_size=4;
stride_size=2;
classSize=2;

layers=[
%     imageInputLayer(inputSize); %Defines the input size
%     
%     convolution2dLayer(FilterSize,InitialNumFilter,'Padding','same'); %,'Stride',ConvolutionStride);   %Input size is the same as output size
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(pool_size,'Stride',stride_size);
%     
%     
%     
%     convolution2dLayer(FilterSize,(1/2)*InitialNumFilter,'Padding','same');   %Input size is the same as output size
%     batchNormalizationLayer
%     reluLayer
%     
%     maxPooling2dLayer(pool_size,'Stride',stride_size);
%     
%     
%     
%     convolution2dLayer(FilterSize,(1/4)*InitialNumFilter,'Padding','same');   %Input size is the same as output size
%     batchNormalizationLayer
%     reluLayer
%     
%     fullyConnectedLayer(classSize)
%     softmaxLayer
%     classificationLayer
    

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



%% Training Options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',16, ...
    'Shuffle','every-epoch', ...
    'ValidationData',CellValidate, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

imageSize = size(readimage(cellDatastore, 1));
augmentedTrainingSet = augmentedImageDatastore(imageSize, CellTrain, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, CellValidate, 'ColorPreprocessing', 'gray2rgb');

%% Training Network
net=trainNetwork(augmentedTrainingSet,layers,options);





