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

%Splitting the data into training and test sets. Training sets will be used to train the model
%and perform 10-fold cross-validation. The test sets will be spit again into 20 further test sets which will be 
%used in an ANOVA to compare the accuracy between the classifiers
cv=cvpartition(length(ULDA_Features(:,1)),'HoldOut',0.33);  %Train: 67%, Test: 33%
index=cv.test;
ULDA_Features_Test=ULDA_Features(index,:);    %Testing data
ULDA_Features=ULDA_Features(~index,:);    %Training data
TestLabels=labels(index);   %Test labels
labelsnew=labels(~index);  %Labels for training


%% ----------------------<Training Classifiers>------------------------
rng('default'); %Sets the random number generator for repeatability
tallrng('default');
%% ---------------------------<Naive Bayes>-----------------------------------

NB_B_Model=fitcnb(FeatureArray,labels);   %Creates a linear model
NB_Model=fitcnb(ULDA_Features,labelsnew);   %Creates a linear model
NB_FS_Model=fitcnb(FS_Features,labels);   %Creates a linear model
NB_FS_ULDA_Model=fitcnb(FS_ULDA_Features,labels);   %Creates a linear model

NB_B_CVModel=crossval(NB_B_Model);    %Cross Validates the model using 10-fold cross validation
NB_CVModel=crossval(NB_Model);    %Cross Validates the model using 10-fold cross validation
NB_FS_CVModel=crossval(NB_FS_Model);    %Cross Validates the model using 10-fold cross validation
NB_FS_ULDA_CVModel=crossval(NB_FS_ULDA_Model);    %Cross Validates the model using 10-fold cross validation

ACC_NB_B=1-kfoldLoss(NB_B_CVModel);   %Determines average accuracy of the lda model
ACC_NB=1-kfoldLoss(NB_CVModel);   %Determines average accuracy of the lda model
ACC_FS_NB=1-kfoldLoss(NB_FS_CVModel);   %Determines average accuracy of the lda model
ACC_FS_ULDA_NB=1-kfoldLoss(NB_FS_ULDA_CVModel);   %Determines average accuracy of the lda model



%% -----------------------------<LDA>-----------------------------------

LDA_B_Model=fitcdiscr(FeatureArray,labels,'discrimtype','linear');   %Creates a linear model
LDA_Model=fitcdiscr(ULDA_Features,labelsnew,'discrimtype','linear');   %Creates a linear model
LDA_FS_Model=fitcdiscr(FS_Features,labels,'discrimtype','linear');   %Creates a linear model
LDA_FS_ULDA_Model=fitcdiscr(FS_ULDA_Features,labels,'discrimtype','linear');   %Creates a linear model

LDA_B_CVModel=crossval(LDA_B_Model);    %Cross Validates the model using 10-fold cross validation
LDA_CVModel=crossval(LDA_Model);    %Cross Validates the model using 10-fold cross validation
LDA_FS_CVModel=crossval(LDA_FS_Model);    %Cross Validates the model using 10-fold cross validation
LDA_FS_ULDA_CVModel=crossval(LDA_FS_ULDA_Model);    %Cross Validates the model using 10-fold cross validation

ACC_LDA_B=1-kfoldLoss(LDA_B_CVModel);   %Determines average accuracy of the lda model
ACC_LDA=1-kfoldLoss(LDA_CVModel);   %Determines average accuracy of the lda model
ACC_FS_LDA=1-kfoldLoss(LDA_FS_CVModel);   %Determines average accuracy of the lda model
ACC_FS_LDA_ULDA=1-kfoldLoss(LDA_FS_ULDA_CVModel);   %Determines average accuracy of the lda model


%% -----------------------------<QDA>----------------------------------
QDA_Model=fitcdiscr(ULDA_Features,labelsnew,'discrimtype','quadratic');   %Creates a quadratic model
QDA_FS_Model=fitcdiscr(FS_Features,labels,'discrimtype','quadratic');   %Creates a quadratic model
QDA_FS_ULDA_Model=fitcdiscr(FS_ULDA_Features,labels,'discrimtype','quadratic');   %Creates a quadratic model
QDA_B_Model=fitcdiscr(FeatureArray,labels,'discrimtype','quadratic');   %Creates a quadratic model

QDA_CVModel=crossval(QDA_Model);    %Cross Validates the model using 10-fold cross validation
QDA_FS_CVModel=crossval(QDA_FS_Model);    %Cross Validates the model using 10-fold cross validation
QDA_FS_ULDA_CVModel=crossval(QDA_FS_ULDA_Model);    %Cross Validates the model using 10-fold cross validation
QDA_B_CVModel=crossval(QDA_B_Model);    %Cross Validates the model using 10-fold cross validation

ACC_QDA=1-kfoldLoss(QDA_CVModel);   %Determines the accuracy of the lda model
ACC_FS_QDA=1-kfoldLoss(QDA_FS_CVModel);   %Determines the accuracy of the lda model
ACC_FS_ULDA_QDA=1-kfoldLoss(QDA_FS_ULDA_CVModel);   %Determines the accuracy of the lda model
ACC_QDA_B=1-kfoldLoss(QDA_B_CVModel);   %Determines the accuracy of the lda model


%% -----------------------------<kNN>----------------------------------
%Finds the hyperparameters (k and distance measure) that minimuze the loss by using the
%automatic hyperparameter optimization and 10-fold cross validation
c=cvpartition(length(ULDA_Features(:,1)),'Kfold',10);
c_B=cvpartition(length(FeatureArray(:,1)),'Kfold',10);

kNN_Optimize=fitcknn(ULDA_Features,labelsnew,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
kNN_B_Optimize=fitcknn(FeatureArray,labels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c_B,...
    'ShowPlots',false,'Verbose',0));
k=kNN_Optimize.NumNeighbors; %Optimal number of neighbours
k_B=kNN_B_Optimize.NumNeighbors; %Optimal number of neighbours

DistMeas=kNN_Optimize.Distance;
DistMeas_B=kNN_B_Optimize.Distance;


%Train the kNN
kNN_Model=fitcknn(ULDA_Features,labelsnew,'NumNeighbors',k,'Distance',DistMeas);    %Trains kNN with optimal hyperparameters
kNN_FS_Model=fitcknn(FS_Features,labels,'NumNeighbors',k,'Distance',DistMeas);    %Trains kNN with optimal hyperparameters
kNN_FS_ULDA_Model=fitcknn(FS_ULDA_Features,labels,'NumNeighbors',k,'Distance',DistMeas);    %Trains kNN with optimal hyperparameters
kNN_B_Model=fitcknn(FeatureArray,labels,'NumNeighbors',k_B,'Distance',DistMeas_B);    %Trains kNN with optimal hyperparameters

kNN_CVModel=crossval(kNN_Model);    %Cross Validates the model using 10-fold cross validation
kNN_FS_CVModel=crossval(kNN_FS_Model);    %Cross Validates the model using 10-fold cross validation
kNN_FS_ULDA_CVModel=crossval(kNN_FS_ULDA_Model);    %Cross Validates the model using 10-fold cross validation
kNN_B_CVModel=crossval(kNN_B_Model);    %Cross Validates the model using 10-fold cross validation

ACC_kNN=1-kfoldLoss(kNN_CVModel);   %Determines the accuracy of the lda model
ACC_FS_kNN=1-kfoldLoss(kNN_FS_CVModel);   %Determines the accuracy of the lda model
ACC_FS_ULDA_kNN=1-kfoldLoss(kNN_FS_ULDA_CVModel);   %Determines the accuracy of the lda model
ACC_B_kNN=1-kfoldLoss(kNN_B_CVModel);   %Determines the accuracy of the lda model


%% ---------------------------<Decision Tree>-------------------------
%Automatically optimizes to find the minimum leaf size hyperparameter using
%10 fold cross validation
DT_Optimize=fitctree(ULDA_Features,labelsnew,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
DT_B_Optimize=fitctree(FeatureArray,labels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c_B,...
    'ShowPlots',false,'Verbose',0));
MinLeaf=DT_Optimize.ModelParameters.MinLeaf;    %Minimum size of leaves
MinLeaf_B=DT_B_Optimize.ModelParameters.MinLeaf;    %Minimum size of leaves

%Train the Decision Tree
DT_Model=fitctree(ULDA_Features,labelsnew,'MinLeafSize',MinLeaf);    %Trains kNN with optimal hyperparameters
DT_FS_Model=fitctree(FS_Features,labels,'MinLeafSize',MinLeaf);    %Trains kNN with optimal hyperparameters
DT_FS_ULDA_Model=fitctree(FS_ULDA_Features,labels,'MinLeafSize',MinLeaf);    %Trains kNN with optimal hyperparameters
DT_B_Model=fitctree(FeatureArray,labels,'MinLeafSize',MinLeaf_B);    %Trains kNN with optimal hyperparameters

DT_CVModel=crossval(DT_Model);    %Cross Validates the model using 10-fold cross validation
DT_FS_CVModel=crossval(DT_FS_Model);    %Cross Validates the model using 10-fold cross validation
DT_FS_ULDA_CVModel=crossval(DT_FS_ULDA_Model);    %Cross Validates the model using 10-fold cross validation
DT_B_CVModel=crossval(DT_B_Model);    %Cross Validates the model using 10-fold cross validation

ACC_DT=1-kfoldLoss(DT_CVModel);   %Determines the accuracy of the DT model
ACC_FS_DT=1-kfoldLoss(DT_FS_CVModel);   %Determines the accuracy of the DT model
ACC_FS_ULDA_DT=1-kfoldLoss(DT_FS_ULDA_CVModel);   %Determines the accuracy of the DT model
ACC_DT_B=1-kfoldLoss(DT_B_CVModel);   %Determines the accuracy of the DT model


%% ------------------------------<SVM>--------------------------------
%Trains a multiclass error-correcting outputs codes using k(k-1)/2 binary
%support vector machine. We also standardize the predictors
%Optimize the hyperparameters
SVM_Optimize=fitcecoc(ULDA_Features,labelsnew,'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
SVM_B_Optimize=fitcecoc(FeatureArray,labels,'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c_B,...
    'ShowPlots',false,'Verbose',0));
BoxCon=table2array(SVM_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,2));
BoxCon_B=table2array(SVM_B_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,2));
KernelScale=table2array(SVM_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,3));
KernelScale_B=table2array(SVM_B_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,3));

t=templateSVM('BoxConstraint',BoxCon,'KernelScale',KernelScale);
t_B=templateSVM('BoxConstraint',BoxCon_B,'KernelScale',KernelScale_B);

SVM_Model=fitcecoc(ULDA_Features,labelsnew,'Learners',t);
SVM_FS_Model=fitcecoc(FS_Features,labels,'Learners',t);
SVM_FS_ULDA_Model=fitcecoc(FS_ULDA_Features,labels,'Learners',t);
SVM_B_Model=fitcecoc(FeatureArray,labels,'Learners',t_B);

SVM_CVModel=crossval(SVM_Model);    %Cross Validates the model using 10-fold cross validation
SVM_FS_CVModel=crossval(SVM_FS_Model);    %Cross Validates the model using 10-fold cross validation
SVM_FS_ULDA_CVModel=crossval(SVM_FS_ULDA_Model);    %Cross Validates the model using 10-fold cross validation
SVM_B_CVModel=crossval(SVM_B_Model);    %Cross Validates the model using 10-fold cross validation

ACC_SVM=1-kfoldLoss(SVM_CVModel);   %Determines the accuracy of the DT model
ACC_FS_SVM=1-kfoldLoss(SVM_FS_CVModel);   %Determines the accuracy of the DT model
ACC_FS_ULDA_SVM=1-kfoldLoss(SVM_FS_ULDA_CVModel);   %Determines the accuracy of the DT model
ACC_SVM_B=1-kfoldLoss(SVM_B_CVModel);   %Determines the accuracy of the DT model



%% ---------------------<Export Matlab Workspace>----------------------
save('Classifiers.mat','NB_B_Model','LDA_B_Model','QDA_B_Model','kNN_B_Model','DT_B_Model','SVM_B_Model');


%% --------------<Plotting 10-Fold Cross Validation Results>--------------
BarLabels=categorical({'NB','LDA','QDA','kNN','DT','SVM'});  %Labels for Bar Graph

Acc=[ACC_NB,ACC_FS_NB,ACC_FS_ULDA_NB;ACC_LDA,ACC_FS_LDA,ACC_FS_LDA_ULDA;...
    ACC_QDA,ACC_FS_QDA,ACC_FS_ULDA_QDA;ACC_kNN,ACC_FS_kNN,ACC_FS_ULDA_kNN;...
    ACC_DT,ACC_FS_DT,ACC_FS_ULDA_DT;ACC_SVM,ACC_FS_SVM,ACC_FS_ULDA_SVM];
figure;
bar(BarLabels,Acc);
legend('ULDA','MRMR','MRMR and ULDA');
xlabel('Classifier');
ylabel('Average Accuracy');
ylim([0.5 1.1]);
title('Accuracy of 6 Classifiers With 3 Pre-Processing Approaches Using 10-fold Cross Validation');


%% -----------------------<ANOVA On Test Data>----------------------------

%The best performance is only with ULDA, so we perform ULDA on the test
%data and then split this data into 20 partitions to check the accuracy of
%each classifier and then perform an ANOVA

%-------<Partition Test Data into 20 Parts and Evaluate Accuracy>----------
NB_Test_ACC=TestAcc(ULDA_Features_Test,TestLabels,NB_Model);
LDA_Test_ACC=TestAcc(ULDA_Features_Test,TestLabels,LDA_Model);
QDA_Test_ACC=TestAcc(ULDA_Features_Test,TestLabels,QDA_Model);
kNN_Test_ACC=TestAcc(ULDA_Features_Test,TestLabels,kNN_Model);
DT_Test_ACC=TestAcc(ULDA_Features_Test,TestLabels,DT_Model);
SVM_Test_ACC=TestAcc(ULDA_Features_Test,TestLabels,SVM_Model);

ACCArray=[NB_Test_ACC',LDA_Test_ACC',QDA_Test_ACC',kNN_Test_ACC',DT_Test_ACC',SVM_Test_ACC'];

%Performing an ANOVA
anova1(ACCArray)

%% ----------------------<Confusion Matrix>-------------------------------
Result_LDA=predict(LDA_Model,ULDA_Features_Test);

NewLabels=[];
PredictLabels=[];
for(i=[1:length(TestLabels)])
    if(TestLabels(i)==1)
        NewLabels{i}='Normal';
        
    else
        NewLabels{i}='Sickle';
    end
end

for(i=[1:length(Result_LDA)])
    if(Result_LDA(i)==1)
        PredictLabels{i}='Normal';
        
    else
        PredictLabels{i}='Sickle';
    end
end



cm=confusionchart(NewLabels,PredictLabels);
cm.Title='Sickle Cell Classification Using ULDA';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
