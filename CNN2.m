clear
clc


%% Read In Grayscale Images

ImageDatasetPath=fullfile('GrayImages'); %Folder containing subfolders for different classes

ImageData=imageDatastore(ImageDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');


%% Split the dataset
[CellTrain,CellValidate]=splitEachLabel(ImageData,0.7,'randomize');


%% Setting layers of CNN
%Setting Parameters
inputSize=[80,80,1];    %Input image is an 80x80 grayscale
FilterSize=3;
ConvolutionStride=1;
InitialNumFilter=8;
pool_size=3;
stride_size=3;
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
    
    convolution2dLayer(FilterSize,InitialNumFilter*2,'Padding','same','Stride',ConvolutionStride);   %Input size is the same as output size
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(pool_size,'Stride',stride_size);
    
    
    %Second 2 convolutional layers
    convolution2dLayer(FilterSize,InitialNumFilter*4,'Padding','same','Stride',ConvolutionStride);   %Input size is the same as output size
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(FilterSize,InitialNumFilter,'Padding','same','Stride',ConvolutionStride);   %Input size is the same as output size
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
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',CellValidate, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

%% Training Network
net=trainNetwork(CellTrain,layers,options);




