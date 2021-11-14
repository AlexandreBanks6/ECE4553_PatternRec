%{
Title: PipeLine1
Author: Alexandre Banks, Christian  Morrel
Description: Extracts features from individual blood cell images, uses features selection,
and dimensionality reduction, and trains/evaluates classifiers
(NaiveBays,LDA,QDA,kNN,parzen,dicision trees, and SVM). There are three
class labels: circular,elongated (sickle cell),other 
Date:November 9th, 2021
Affiliation: University of New Brunswick, ECE 4553

%}

clear 
clc

%% Reading Data

circularcell_files=dir('circular\*.jpg');  %dir gives a list of all the jpg files in the database of circular cells
elongatedcell_files=dir('elongated\*.jpg');  %dir gives a list of all the jpg files in the database of circular cells
othercell_files=dir('other\*.jpg');  %dir gives a list of all the jpg files in the database of circular cells

circ_im=readimagefiles(circularcell_files,'circular'); %Circular blood cell images
elong_im=readimagefiles(elongatedcell_files,'elongated'); %Circular blood cell images
other_im=readimagefiles(othercell_files,'other'); %Circular blood cell images

%Creating labels (1==circular, 2==elongated, 3==other)
labels=[ones(1,length(circ_im)),2*ones(1,length(elong_im)),1*ones(1,length(other_im))];
labels=labels';
%Combining datasets
ImData=[circ_im,elong_im,other_im];



%% Feature Extraction 
%(We might want to look into normalizing by number of pixels)!!!
ImGray=ConvRGB_to_GRAY(ImData); %Extracts the grayscale images

[FeatureArray, FeatNames] = extractCellFeatures(ImGray);

% %----------------------<Binarize each image>----------------
% %Returns cell array containing all the binary images
% BW_Dat=binarizeimage(ImGray);   
% %---------------------<Only Cells Extracted>--------------------
% %Complements images, fills holes, and selects the central cell
% BW_Dat_Ref_Filled=onlycell(BW_Dat);
% 
% %Complements images, and selects the central cell
% BW_Dat_Ref=ComplementAndExtract(BW_Dat);
% 
% %---------------<Colour Images with only cells>--------------
% %ImData_Ref=onlycolourcell(ImData,BW_Dat_Ref_Filled);
% %---------------<Only Grayscale Image With Cell>-------------
% ImGray=extractGray(ImGray,BW_Dat_Ref_Filled);
% 
% %---------------<Setting Parameters>-----------------------------
% nImages=length(BW_Dat_Ref_Filled);   %Finds the number of images that we are analyzing
% 
% 
% %Computing Perimeter
% PerimVec=zeros(1,nImages);
% for(i=[1:nImages])
%         %NumPix=findpixnum(BW_Dat_Ref_Filled{i});           %We normalize the perimeter by the total number of pixels
%         perim=bwperim(BW_Dat_Ref_Filled{i});              %Determines perimeter elements (1==perimeter)
%         PerimVec(i)=sum(perim(:)==1); %Finds the number of pixels in the perimeter
% end
% 
% %Computing Area of Cells
% AreaVec=zeros(1,nImages);
% for(i=[1:nImages])
%         %NumPix=findpixnum(BW_Dat_Ref_Filled{i});           %We normalize the area by the total number of pixels
%         AreaVec(i)=bwarea(BW_Dat_Ref_Filled{i}); %Approximate number of pixels in the cell
% end
% 
% %Finding Circularity
% CircularityVec=4*pi.*(AreaVec./(PerimVec.^2)); %Circularity calculation
% 
% %Computing Minimum Diameter
% MinDiamVec=zeros(1,nImages);
% for(i=[1:nImages])
%         %NumPix=findpixnum(BW_Dat_Ref_Filled{i});           %We normalize the min diameter by the total number of pixels
%         out=bwferet(BW_Dat_Ref_Filled{i},'MinFeretProperties'); %Finds the minimum properties of the image
%         MinDiamVec(i)=out.MinDiameter(1); %Approximate minimum diameter of cells
% end
% 
% %Computing Maximum Diameter
% MaxDiamVec=zeros(1,nImages);
% for(i=[1:nImages])
%         %NumPix=findpixnum(BW_Dat_Ref_Filled{i});           %We normalize the max diameter by the total number of pixels
%         out=bwferet(BW_Dat_Ref_Filled{i},'MaxFeretProperties'); %Finds the maximum properties of the image
%         MaxDiamVec(i)=out.MaxDiameter(1); %Approximate maximum diameter of cells
% end
% 
% %Calculating Elipticity
% ElipVec=zeros(1,nImages);
% for(i=[1:nImages])
%         ElipVec(i)=elipticityfun(MinDiamVec(i),MaxDiamVec(i)); %Approximate maximum diameter of cells
% end
% 
% %Finding Convex Hull Area
% ConvexHullVec=zeros(1,nImages);
% for(i=[1:nImages])
%         Val=regionprops(BW_Dat_Ref{i},'ConvexArea');
%         ConvexHullVec(i)=Val.ConvexArea;
% end
% 
% %Computes the eccentricity of the ellipse where the values is between 0 and
% %1 (0=circle, 1=line)
% EccentricityVec=zeros(1,nImages);
% for(i=[1:nImages])
%     Val=regionprops(BW_Dat_Ref_Filled{i},'Eccentricity');    
%     EccentricityVec(i)=Val.Eccentricity;
% end
% 
% %Computes the equivalent diameter of the images, where the equivalent
% %diameter is the diameter of a circle with the same are aas the region in
% %the image
% EquivDiamVec=zeros(1,nImages);
% for(i=[1:nImages])
%         Val=regionprops(BW_Dat_Ref{i},'EquivDiameter');
%         EquivDiamVec(i)=Val.EquivDiameter;
% end
% 
% 
% 
% %Computes solidity as the cell area/convex area
% SolidityVec=zeros(1,nImages);
% for(i=[1:nImages])
%         Val=regionprops(BW_Dat_Ref{i},'Solidity');
%         SolidityVec(i)=Val.Solidity;
% end
% 
% %Computes the texture of the images
% %(Standard Deviation of Grayscale values)
% TextureVec=zeros(1,nImages);
% for(i=[1:nImages])
%         X=ImGray{i}; %Grascale Image Matrix
%         TextureVec(i)=std(im2double(X),0,'all');
% end
% 
% %Symmetry (Calculated as mean squared error between image and flipped
% %image)
% SymmetryVec=zeros(1,nImages);
% for(i=[1:nImages])
%         X=ImGray{i};
%         SymmetryVec(i)=immse(X,fliplr(X));    %Calculates symmetry
% end
% %Array of features
% FeatureArray=[PerimVec',AreaVec',CircularityVec',MinDiamVec',MaxDiamVec',ElipVec',ConvexHullVec',...
%     EccentricityVec',EquivDiamVec',SolidityVec',TextureVec',SymmetryVec'];
% FeatNames={'Perim','Area','Circularity','MinDiam','MaxDiam','Elipticity','ConvexArea',...
%     'Eccentricity','EquivalentDiam','Solidity','Texture','Symmetry'};
%% Sequential Feature Selection

%Function defining criterion used to select features (this is using sum of squares from line)
% fun=@(XT,yT,Xt,yt) (sum(~strcmp(yt,classify(Xt,XT,yT,'linear')))); 
% c=cvpartition(labels,'k',10); %10-fold cross-validation
% [result,history]=sequentialfs(fun,FeatureArray,labels,'cv',c);%Perform sequential feature selection
% 
% Feature_Selected=FeatureArray(:,result);
% SelectedFeatureName_Ordered=[]; %Initializes array of feature names that will be ordered
% 
% Feature_Selected_Ordered=zeros(length(FeatureArray(:,1)),sum(result==1)); %Vector which contains the features ordered
% indexvec1=zeros(1,length(FeatureArray(1,:)));
% indexvec2=zeros(1,length(FeatureArray(1,:)));
% histresult=history.In;
% %ordered from most to least relevant of the selected feature
% for(i=[1:sum(result==1)])
%     indexvec2=histresult(i,:);
%     SelectedFeatureName_Ordered=[SelectedFeatureName_Ordered,FeatNames(find((indexvec2-indexvec1)==1))];
%     Feature_Selected_Ordered(:,i)=FeatureArray(:,find((indexvec2-indexvec1)==1));
%     indexvec1=indexvec2;    
% end

%% Feature Ranking Using MRMR Algorithm

[idx,score]=fscmrmr(FeatureArray,labels);
Features_mrmr=FeatureArray(:,idx);
idxScores = score(idx);

% Plot of scores
figure()
bar(idxScores)

% Find number of features to include
featurePerc = 80;   % Percentage of feature score to retain
currentScore = 0;
totalScore = sum(score);    % Total score of all features
for i = 1:length(score)
    currentScore = currentScore + idxScores(i);
    if currentScore/totalScore > featurePerc/100
        numFeats = i;   % Number of features to use after feature selection
        break;
    end
end

FS_Features = Features_mrmr(:, 1:numFeats); % Features after feature selection



%% Using ULDA For Dimensionality Reduction
PercGoal=95;    %95 percent of total variance explained by projected data
[ULDA_Features,explained,ProjDatUnCleaned]=ULDA(FeatureArray,labels,PercGoal);

[FS_ULDA_Features,FS_explained,FS_ProjDatUnCleaned]=ULDA(FS_Features,labels,PercGoal);




%% ----------------------<Training Classifiers>------------------------
rng('default'); %Sets the random number generator for repeatability
tallrng('default');
%% ---------------------------<Naive Bayes>-----------------------------------

NB_Model=fitcnb(ULDA_Features,labels);   %Creates a linear model
NB_FS_Model=fitcnb(FS_Features,labels);   %Creates a linear model
NB_FS_ULDA_Model=fitcnb(FS_ULDA_Features,labels);   %Creates a linear model


NB_CVModel=crossval(NB_Model);    %Cross Validates the model using 10-fold cross validation
NB_FS_CVModel=crossval(NB_FS_Model);    %Cross Validates the model using 10-fold cross validation
NB_FS_ULDA_CVModel=crossval(NB_FS_ULDA_Model);    %Cross Validates the model using 10-fold cross validation


ACC_NB=1-kfoldLoss(NB_CVModel);   %Determines average accuracy of the lda model
ACC_FS_NB=1-kfoldLoss(NB_FS_CVModel);   %Determines average accuracy of the lda model
ACC_FS_ULDA_NB=1-kfoldLoss(NB_FS_ULDA_CVModel);   %Determines average accuracy of the lda model



%% -----------------------------<LDA>-----------------------------------

LDA_Model = fitcdiscr(FeatureArray,labels,'discrimtype','linear');   %Creates a linear model
LDA_Model_ULDA=fitcdiscr(ULDA_Features,labels,'discrimtype','linear');   %Creates a linear model
LDA_FS_Model=fitcdiscr(FS_Features,labels,'discrimtype','linear');   %Creates a linear model
LDA_FS_ULDA_Model=fitcdiscr(FS_ULDA_Features,labels,'discrimtype','linear');   %Creates a linear model

LDA_CVModel=crossval(LDA_Model);    %Cross Validates the model using 10-fold cross validation
LDA_CVModel_ULDA=crossval(LDA_Model_ULDA);    %Cross Validates the model using 10-fold cross validation
LDA_FS_CVModel=crossval(LDA_FS_Model);    %Cross Validates the model using 10-fold cross validation
LDA_FS_ULDA_CVModel=crossval(LDA_FS_ULDA_Model);    %Cross Validates the model using 10-fold cross validation

ACC_LDA = 1 - kfoldLoss(LDA_CVModel);
ACC_LDA_ULDA=1-kfoldLoss(LDA_CVModel_ULDA);   %Determines average accuracy of the lda model
ACC_FS_LDA=1-kfoldLoss(LDA_FS_CVModel);   %Determines average accuracy of the lda model
ACC_FS_LDA_ULDA=1-kfoldLoss(LDA_FS_ULDA_CVModel);   %Determines average accuracy of the lda model


%% -----------------------------<QDA>----------------------------------
QDA_Model=fitcdiscr(ULDA_Features,labels,'discrimtype','quadratic');   %Creates a quadratic model
QDA_FS_Model=fitcdiscr(FS_Features,labels,'discrimtype','quadratic');   %Creates a quadratic model
QDA_FS_ULDA_Model=fitcdiscr(FS_ULDA_Features,labels,'discrimtype','quadratic');   %Creates a quadratic model

QDA_CVModel=crossval(QDA_Model);    %Cross Validates the model using 10-fold cross validation
QDA_FS_CVModel=crossval(QDA_FS_Model);    %Cross Validates the model using 10-fold cross validation
QDA_FS_ULDA_CVModel=crossval(QDA_FS_ULDA_Model);    %Cross Validates the model using 10-fold cross validation

ACC_QDA=1-kfoldLoss(QDA_CVModel);   %Determines the accuracy of the lda model
ACC_FS_QDA=1-kfoldLoss(QDA_FS_CVModel);   %Determines the accuracy of the lda model
ACC_FS_ULDA_QDA=1-kfoldLoss(QDA_FS_ULDA_CVModel);   %Determines the accuracy of the lda model


%% -----------------------------<kNN>----------------------------------
%Finds the hyperparameters (k and distance measure) that minimuze the loss by using the
%automatic hyperparameter optimization and 10-fold cross validation
c=cvpartition(length(ULDA_Features(:,1)),'Kfold',10);
kNN_Optimize=fitcknn(ULDA_Features,labels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
k=kNN_Optimize.NumNeighbors; %Optimal number of neighbours
DistMeas=kNN_Optimize.Distance;

%Train the kNN
kNN_Model=fitcknn(ULDA_Features,labels,'NumNeighbors',k,'Distance',DistMeas);    %Trains kNN with optimal hyperparameters
kNN_FS_Model=fitcknn(FS_Features,labels,'NumNeighbors',k,'Distance',DistMeas);    %Trains kNN with optimal hyperparameters
kNN_FS_ULDA_Model=fitcknn(FS_ULDA_Features,labels,'NumNeighbors',k,'Distance',DistMeas);    %Trains kNN with optimal hyperparameters

kNN_CVModel=crossval(kNN_Model);    %Cross Validates the model using 10-fold cross validation
kNN_FS_CVModel=crossval(kNN_FS_Model);    %Cross Validates the model using 10-fold cross validation
kNN_FS_ULDA_CVModel=crossval(kNN_FS_ULDA_Model);    %Cross Validates the model using 10-fold cross validation

ACC_kNN=1-kfoldLoss(kNN_CVModel);   %Determines the accuracy of the lda model
ACC_FS_kNN=1-kfoldLoss(kNN_FS_CVModel);   %Determines the accuracy of the lda model
ACC_FS_ULDA_kNN=1-kfoldLoss(kNN_FS_ULDA_CVModel);   %Determines the accuracy of the lda model


%% ---------------------------<Decision Tree>-------------------------
%Automatically optimizes to find the minimum leaf size hyperparameter using
%10 fold cross validation
DT_Optimize=fitctree(ULDA_Features,labels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
MinLeaf=DT_Optimize.ModelParameters.MinLeaf;    %Minimum size of leaves

%Train the Decision Tree
DT_Model=fitctree(ULDA_Features,labels,'MinLeafSize',MinLeaf);    %Trains kNN with optimal hyperparameters
DT_FS_Model=fitctree(FS_Features,labels,'MinLeafSize',MinLeaf);    %Trains kNN with optimal hyperparameters
DT_FS_ULDA_Model=fitctree(FS_ULDA_Features,labels,'MinLeafSize',MinLeaf);    %Trains kNN with optimal hyperparameters

DT_CVModel=crossval(DT_Model);    %Cross Validates the model using 10-fold cross validation
DT_FS_CVModel=crossval(DT_FS_Model);    %Cross Validates the model using 10-fold cross validation
DT_FS_ULDA_CVModel=crossval(DT_FS_ULDA_Model);    %Cross Validates the model using 10-fold cross validation

ACC_DT=1-kfoldLoss(DT_CVModel);   %Determines the accuracy of the DT model
ACC_FS_DT=1-kfoldLoss(DT_FS_CVModel);   %Determines the accuracy of the DT model
ACC_FS_ULDA_DT=1-kfoldLoss(DT_FS_ULDA_CVModel);   %Determines the accuracy of the DT model


%% ------------------------------<SVM>--------------------------------
%Trains a multiclass error-correcting outputs codes using k(k-1)/2 binary
%support vector machine. We also standardize the predictors
%Optimize the hyperparameters
SVM_Optimize=fitcecoc(ULDA_Features,labels,'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
BoxCon=table2array(SVM_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,2));
KernelScale=table2array(SVM_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,3));

t=templateSVM('BoxConstraint',BoxCon,'KernelScale',KernelScale);
SVM_Model=fitcecoc(ULDA_Features,labels,'Learners',t);
SVM_FS_Model=fitcecoc(FS_Features,labels,'Learners',t);
SVM_FS_ULDA_Model=fitcecoc(FS_ULDA_Features,labels,'Learners',t);

SVM_CVModel=crossval(SVM_Model);    %Cross Validates the model using 10-fold cross validation
SVM_FS_CVModel=crossval(SVM_FS_Model);    %Cross Validates the model using 10-fold cross validation
SVM_FS_ULDA_CVModel=crossval(SVM_FS_ULDA_Model);    %Cross Validates the model using 10-fold cross validation

ACC_SVM=1-kfoldLoss(SVM_CVModel);   %Determines the accuracy of the DT model
ACC_FS_SVM=1-kfoldLoss(SVM_FS_CVModel);   %Determines the accuracy of the DT model
ACC_FS_ULDA_SVM=1-kfoldLoss(SVM_FS_ULDA_CVModel);   %Determines the accuracy of the DT model



%% ---------------------<Export Matlab Workspace>----------------------
save('Classifiers.mat','NB_Model','LDA_Model','QDA_Model','kNN_Model','DT_Model','SVM_Model');