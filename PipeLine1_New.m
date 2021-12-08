%{
Title: PipeLine1_New
Author: Alexandre Banks, Christian  Morrel
Date:November 9th, 2021
Affiliation: University of New Brunswick, ECE 4553
Description: Extracts features from individual blood cell images, uses features selection,
and dimensionality reduction, and trains/evaluates classifiers
(NaiveBays,LDA,QDA,kNN,parzen,dicision trees, and SVM). Two types of
classification algorithms (sequential feature selection and minimum
redundancy maximum relavance) are tested with different combinations with two
dimensionality reduction approaches (principal component analysis and
fisher's linear discriminant analysis). The pre-processing approach that
returns the best accuracies for the given classifiers is selected and
further analysis is conducted on this pipeline.
There are two class labels: elongated,and heatlhy (healthy contains both circular and abnormal shaped cells) 
This script also plots various performance metrics of the classifiers
including a confusion matrix, an ANOVA comparing the classifiers on testing
data, and a plot of the classifier accuracies for various pre-processing
approaches.



%}

clear 
clc

%% Reading Data

circularcell_files=dir('circular\*.jpg');  %dir gives a list of all the jpg files in the database of circular cells
elongatedcell_files=dir('elongated\*.jpg');  %dir gives a list of all the jpg files in the database of elongated cells
othercell_files=dir('other\*.jpg');  %dir gives a list of all the jpg files in the database of other cells

circ_im=readimagefiles(circularcell_files,'circular'); %Circular blood cell images
elong_im=readimagefiles(elongatedcell_files,'elongated'); %Elongated blood cell images
other_im=readimagefiles(othercell_files,'other'); %Other blood cell images

%Labels: 1=circular and other, 2=elongated
labels=[ones(1,length(circ_im)),2*ones(1,length(elong_im)),1*ones(1,length(other_im))]; 
labels=labels';

%Combining datasets
ImData=[circ_im,elong_im,other_im];


%% Feature Extraction 
ImGray=ConvRGB_to_GRAY(ImData); %Extracts the grayscale images

[FeatureArray, FeatNames] = extractCellFeatures(ImGray);    %Extracting the features for each image
%FeatureArray = array of features from feature extraction
%FeatNames = array of the corresponding feature names

%% Sequential Feature Selection

%Function defining criterion used to select features (this is using sum of squares)
fun=@(XT,yT,Xt,yt) (sum(~strcmp(yt,classify(Xt,XT,yT,'linear')))); 
c=cvpartition(labels,'k',10); %10-fold cross-validation

[result,history]=sequentialfs(fun,FeatureArray,labels,'cv',c);%Perform sequential feature selection

Feature_Selected=FeatureArray(:,result);    %Array of selected features

SelectedFeatureName_Ordered=[]; %Initializes array of feature names that will be ordered

SFS_Features=zeros(length(FeatureArray(:,1)),sum(result==1)); %Vector which contains the features ordered

indexvec1=zeros(1,length(FeatureArray(1,:)));   %Used to store a vector of logical values which indicate features kept on each iteration of the SFS
indexvec2=zeros(1,length(FeatureArray(1,:)));   %Stores the past array of logical values (from indexvec2)
histresult=history.In;
%ordered from most to least relevant of the selected feature
for(i=[1:sum(result==1)])
    indexvec2=histresult(i,:);  %Index of logical values of the features kept
    %Names of features kept
    SelectedFeatureName_Ordered=[SelectedFeatureName_Ordered,FeatNames(find((indexvec2-indexvec1)==1))];
    
    %SFS Features
    SFS_Features(:,i)=FeatureArray(:,find((indexvec2-indexvec1)==1));
    indexvec1=indexvec2;    
end

%% Feature Ranking Using MRMR Algorithm

[idx,score]=fscmrmr(FeatureArray,labels);   %Returns the index of features ordered and their score

Features_mrmr=FeatureArray(:,idx);  %Features re-ordered by mRMR

idxScores = score(idx);

% Plot of scores
% figure()
% bar(idxScores)

% Find number of features to include
featurePerc = 89;   % Percentage of feature score to retain
currentScore = 0;   %Counter which tracks how much score is explained so far
totalScore = sum(score);    % Total score of all features
for i = 1:length(score) %Loops for the length of the score vector (number of features)
    currentScore = currentScore + idxScores(i); %Increments the current score
    if currentScore/totalScore > featurePerc/100    %If the cumulative score of the features retained exceeds 89% then we exit
        numFeats = i;   % Number of features to use after feature selection
        break;
    end
end

mRMR_Features = Features_mrmr(:, 1:numFeats); % Features after feature selection
mRMR_Feature_Names=FeatNames(idx(1:numFeats));  %Names of features
mRMR_Features_Percentage=100*idxScores(1:numFeats)/totalScore; %The score of the features kept (in percentage)


%% Using ULDA For Dimensionality Reduction
%Uses custom-built fisher's LDA function to maximize between-class variance

PercGoal=95;    %95 percent of total variance explained by projected data
%mRMR Reduced Features

%ULDA_mRMR is the reduced features with mRMR pre-processing
[ULDA_mRMR,explained,ProjDatUnCleaned,ULDA_Weight]=ULDA(mRMR_Features,labels,PercGoal);

%SFS Reduced Features
[ULDA_SFS,FS_explained,FS_ProjDatUnCleaned, ~]=ULDA(SFS_Features,labels,PercGoal);


%% PCA For Dimensionality Reduction
%mRMR Dataset
[Coeff_mRMR_PCA,Proj_mRMR_PCA,latent_mRMR_PCA,~,explained_mRMR_PCA,Mu1]=pca(mRMR_Features,'Algorithm','eig'); 
%Coeff_mRMR=principal component coefficients, explained_mRMR_PCA=representation of data in principal component domainexplained1=percentage of total variance explained
ind1=VarExpInd(explained_mRMR_PCA,95); %Returns index of last principal componet needed to explain up to 95% of the total variance
PCA_mRMR=Proj_mRMR_PCA(:,[1:ind1]); %The PCA Projection



%SFS Dataset
[Coeff_SFS_PCA,Proj_SFS_PCA,latent_SFS_PCA,~,explained_SFS_PCA,Mu2]=pca(SFS_Features,'Algorithm','eig'); 
%Coeff_SFS=principal component coefficients, explained_SFS_PCA=representation of data in principal component domainexplained1=percentage of total variance explained
ind2=VarExpInd(explained_SFS_PCA,95); %Returns index of last principal componet needed to explain up to 95% of the total variance
PCA_SFS=Proj_mRMR_PCA(:,[1:ind2]); %The PCA Projection




%% 
%Splitting the data into training and test sets. Training sets will be used to train the model
%and perform 10-fold cross-validation. The test sets will be spit again into 10 further test sets which will be 
%used in an ANOVA to compare the accuracy between the classifiers
cv=cvpartition(length(PCA_mRMR(:,1)),'HoldOut',0.33);  %Train: 67%, Test: 33%
index=cv.test;

%Test Data
OriginalImages_Test=ImData(index);  %Original images used for plotting results

ULDA_mRMR_Test=ULDA_mRMR(index,:);  %ULDA with mRMR test data
ULDA_SFS_Test=ULDA_SFS(index,:);    %ULDA with SFS test data
PCA_mRMR_Test=PCA_mRMR(index,:);    %PCA with mRMR test data
PCA_SFS_Test=PCA_SFS(index,:);      %PCA with SFS test data

%Training Data
ULDA_mRMR_Train=ULDA_mRMR(~index,:);    %ULDA with mRMR training data
ULDA_SFS_Train=ULDA_SFS(~index,:);      %ULDA with SFS training data
PCA_mRMR_Train=PCA_mRMR(~index,:);      %PCA with mRMR training data
PCA_SFS_Train=PCA_SFS(~index,:);        %PCA with SFS training data


TestLabels=labels(index);   %Test labels
TrainLabels=labels(~index);  %Labels for training

%% Distribution of Data with ULDA and mRMR

% % Colour matrix
[sickleRows, ~] = find(labels == 2);    % Find rows that correspond to sickle
rgb = zeros(height(mRMR_Features), 3);
rgb(sickleRows, 1) = 1; % Set sickle cell points to red
% 
% % MRMR
% figure()
% scatter(mRMR_Features(:, 1), mRMR_Features(:, 2), [], rgb)
% grid on
% xlabel('Minimum Diameter')
% ylabel('Circularity')
% title('Scatter Plot of Two Highest Ranked Features for Individual Blood Cell Images')
% legend('Elongated')
% 
% ULDA
figure()    %Init figure
scatter(1:length(ULDA_mRMR), ULDA_mRMR, [], rgb)    %Scatter plot of  ULDA feature space
grid on
%Setting title handles
xlabel('Observation Number')
ylabel('LDA Feature 1')
title('Scatter Plot of ULDA Feature Space with mRMR')
legend('Elongated')



%% ----------------------<Training Classifiers>------------------------
rng('default'); %Sets the random number generator for repeatability
tallrng('default');
%{
Below we train 6 classifieres (Naive Bayes, LDA, QDA, kNN, decision trees,
and SVM) on the four pre-processing approaches (ULDA with mRMR, ULDA with
SFS, PCA with mRMR, and PCA with SFS)
%}
%% ---------------------------<Naive Bayes>-----------------------------------

NB_ULDA_mRMR_Model=fitcnb(ULDA_mRMR_Train,TrainLabels);   %Creates a linear model
NB_ULDA_SFS_Model=fitcnb(ULDA_SFS_Train,TrainLabels);   %Creates a linear model
NB_PCA_mRMR_Model=fitcnb(PCA_mRMR_Train,TrainLabels);   %Creates a linear model
NB_PCA_SFS_Model=fitcnb(PCA_SFS_Train,TrainLabels);   %Creates a linear model

% !! crossval() uses 10-fold cross validation by default
ACC_NB_ULDA_mRMR=1-kfoldLoss(crossval(NB_ULDA_mRMR_Model));   %Determines average accuracy of the lda model
ACC_NB_ULDA_SFS=1-kfoldLoss(crossval(NB_ULDA_SFS_Model));   %Determines average accuracy of the lda model
ACC_NB_PCA_mRMR=1-kfoldLoss(crossval(NB_PCA_mRMR_Model));   %Determines average accuracy of the lda model
ACC_NB_PCA_SFS=1-kfoldLoss(crossval(NB_PCA_SFS_Model));   %Determines average accuracy of the lda model



%% -----------------------------<LDA>-----------------------------------

LDA_ULDA_mRMR_Model=fitcdiscr(ULDA_mRMR_Train,TrainLabels,'discrimtype','linear');   %Creates an LDA model
LDA_ULDA_SFS_Model=fitcdiscr(ULDA_SFS_Train,TrainLabels,'discrimtype','linear');   %Creates an LDA model
LDA_PCA_mRMR_Model=fitcdiscr(PCA_mRMR_Train,TrainLabels,'discrimtype','linear');   %Creates an LDA model
LDA_PCA_SFS_Model=fitcdiscr(PCA_SFS_Train,TrainLabels,'discrimtype','linear');   %Creates an LDA model


ACC_LDA_ULDA_mRMR=1-kfoldLoss(crossval(LDA_ULDA_mRMR_Model));   %Determines average accuracy of the lda model
ACC_LDA_ULDA_SFS=1-kfoldLoss(crossval(LDA_ULDA_SFS_Model));   %Determines average accuracy of the lda model
ACC_LDA_PCA_mRMR=1-kfoldLoss(crossval(LDA_PCA_mRMR_Model));   %Determines average accuracy of the lda model
ACC_LDA_PCA_SFS=1-kfoldLoss(crossval(LDA_PCA_SFS_Model));   %Determines average accuracy of the lda model


%% -----------------------------<QDA>----------------------------------
% ULDA_mRMR_Train
% ULDA_SFS_Train
% PCA_mRMR_Train
% PCA_SFS_Train

QDA_ULDA_mRMR_model=fitcdiscr(ULDA_mRMR_Train,TrainLabels,'discrimtype','quadratic');   %Creates a quadratic model
QDA_ULDA_SFS_model=fitcdiscr(ULDA_SFS_Train,TrainLabels,'discrimtype','quadratic');   %Creates a quadratic model
QDA_PCA_mRMR_Model=fitcdiscr(PCA_mRMR_Train,TrainLabels,'discrimtype','quadratic');   %Creates a quadratic model
QDA_PCA_SFS_Model=fitcdiscr(PCA_SFS_Train,TrainLabels,'discrimtype','quadratic');   %Creates a quadratic model


ACC_QDA_ULDA_mRMR=1-kfoldLoss(crossval(QDA_ULDA_mRMR_model));   %Determines the accuracy of the qda model
ACC_QDA_ULDA_SFS=1-kfoldLoss(crossval(QDA_ULDA_SFS_model));   %Determines the accuracy of the qda model
ACC_QDA_PCA_mRMR=1-kfoldLoss(crossval(QDA_PCA_mRMR_Model));   %Determines the accuracy of the qda model
ACC_QDA_PCA_SFS=1-kfoldLoss(crossval(QDA_PCA_SFS_Model));   %Determines the accuracy of the qda model


%% -----------------------------<kNN>----------------------------------


%~~~~~~~~~~~~~~~~~~~~~~~~~~~<Optimization>~~~~~~~~~~~~~~~~~~~~~~~~~~
%Finds the hyperparameters (k and distance measure) that minimuze the loss by using the
%automatic hyperparameter optimization and 10-fold cross validation


c=cvpartition(length(ULDA_mRMR_Train(:,1)),'Kfold',10); %Partition data for optimization


%First Optimization (used for ULDA/mRMR data)
kNN_Optimize=fitcknn(ULDA_mRMR_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
k1=kNN_Optimize.NumNeighbors; %Optimal number of neighbours
DistMeas1=kNN_Optimize.Distance;

%Second Optimization (used for ULDA/SFS data)
kNN_Optimize=fitcknn(ULDA_SFS_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
k2=kNN_Optimize.NumNeighbors; %Optimal number of neighbours
DistMeas2=kNN_Optimize.Distance;

%Third Optimization (used for PCA/mRMR data)
kNN_Optimize=fitcknn(PCA_mRMR_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
k3=kNN_Optimize.NumNeighbors; %Optimal number of neighbours
DistMeas3=kNN_Optimize.Distance;

%Fourth Optimization (used for PCA/mRMR data)
kNN_Optimize=fitcknn(PCA_SFS_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
k4=kNN_Optimize.NumNeighbors; %Optimal number of neighbours
DistMeas4=kNN_Optimize.Distance;



%Train the kNN

kNN_ULDA_mRMR_model=fitcknn(ULDA_mRMR_Train,TrainLabels,'NumNeighbors',k1,'Distance',DistMeas1);   %Creates a kNN model
kNN_ULDA_SFS_model=fitcknn(ULDA_SFS_Train,TrainLabels,'NumNeighbors',k2,'Distance',DistMeas2);   %Creates a kNN model
kNN_PCA_mRMR_Model=fitcknn(PCA_mRMR_Train,TrainLabels,'NumNeighbors',k3,'Distance',DistMeas3);   %Creates a kNN model
kNN_PCA_SFS_Model=fitcknn(PCA_SFS_Train,TrainLabels,'NumNeighbors',k4,'Distance',DistMeas4);   %Creates a kNN model


ACC_kNN_ULDA_mRMR=1-kfoldLoss(crossval(kNN_ULDA_mRMR_model));   %Determines the accuracy of the kNN model
ACC_kNN_ULDA_SFS=1-kfoldLoss(crossval(kNN_ULDA_SFS_model));   %Determines the accuracy of the kNN model
ACC_kNN_PCA_mRMR=1-kfoldLoss(crossval(kNN_PCA_mRMR_Model));   %Determines the accuracy of the kNN model
ACC_kNN_PCA_SFS=1-kfoldLoss(crossval(kNN_PCA_SFS_Model));   %Determines the accuracy of the kNN model



%% ---------------------------<Decision Tree>-------------------------
%Automatically optimizes to find the minimum leaf size hyperparameter using
%10 fold cross validation

%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~<Optimization>~~~~~~~~~~~~~~~~~~~~~~~~~~~
%First optimization (used for ULDA/mRMR data)
DT_Optimize=fitctree(ULDA_mRMR_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
MinLeaf1=DT_Optimize.ModelParameters.MinLeaf;    %Minimum size of leaves


%Second optimization (used for ULDA/SFS data)
DT_Optimize=fitctree(ULDA_SFS_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
MinLeaf2=DT_Optimize.ModelParameters.MinLeaf;    %Minimum size of leaves


%Third optimization (used for PCA/mRMR data)
DT_Optimize=fitctree(PCA_mRMR_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
MinLeaf3=DT_Optimize.ModelParameters.MinLeaf;    %Minimum size of leaves


%Fourth optimization (used for PCA/SFS data)
DT_Optimize=fitctree(PCA_SFS_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
MinLeaf4=DT_Optimize.ModelParameters.MinLeaf;    %Minimum size of leaves



DT_ULDA_mRMR_model=fitctree(ULDA_mRMR_Train,TrainLabels,'MinLeafSize',MinLeaf1);   %Creates a decision tree model
DT_ULDA_SFS_model=fitctree(ULDA_SFS_Train,TrainLabels,'MinLeafSize',MinLeaf2);   %Creates a decision tree model
DT_PCA_mRMR_Model=fitctree(PCA_mRMR_Train,TrainLabels,'MinLeafSize',MinLeaf3);   %Creates a decision tree model
DT_PCA_SFS_Model=fitctree(PCA_SFS_Train,TrainLabels,'MinLeafSize',MinLeaf4);   %Creates a decision tree model


ACC_DT_ULDA_mRMR=1-kfoldLoss(crossval(DT_ULDA_mRMR_model));   %Determines the accuracy of the decision tree model
ACC_DT_ULDA_SFS=1-kfoldLoss(crossval(DT_ULDA_SFS_model));   %Determines the accuracy of the decision tree model
ACC_DT_PCA_mRMR=1-kfoldLoss(crossval(DT_PCA_mRMR_Model));   %Determines the accuracy of the decision tree model
ACC_DT_PCA_SFS=1-kfoldLoss(crossval(DT_PCA_SFS_Model));   %Determines the accuracy of the decision tree model



%% ------------------------------<SVM>--------------------------------
%Trains a multiclass error-correcting outputs codes using k(k-1)/2 binary
%support vector machine. We also standardize the predictors


%~~~~~~~~~~~~~~~~~~~~~~Optimize the hyperparameters~~~~~~~~~~~~~~~

%First optimization (used for ULDA/mRMR data)
SVM_Optimize1=fitcecoc(ULDA_mRMR_Train,TrainLabels,'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));

BoxCon1=table2array(SVM_Optimize1.HyperparameterOptimizationResults.XAtMinObjective(1,2));
KernelScale1=table2array(SVM_Optimize1.HyperparameterOptimizationResults.XAtMinObjective(1,3));

t1=templateSVM('BoxConstraint',BoxCon1,'KernelScale',KernelScale1);


%Second optimization (used for ULDA/SFS data)
SVM_Optimize2=fitcecoc(ULDA_SFS_Train,TrainLabels,'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
BoxCon2=table2array(SVM_Optimize2.HyperparameterOptimizationResults.XAtMinObjective(1,2));
KernelScale2=table2array(SVM_Optimize2.HyperparameterOptimizationResults.XAtMinObjective(1,3));
t2=templateSVM('BoxConstraint',BoxCon2,'KernelScale',KernelScale2);


%Third optimization (used for PCA/mRMR data)
SVM_Optimize3=fitcecoc(PCA_mRMR_Train,TrainLabels,'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
BoxCon3=table2array(SVM_Optimize3.HyperparameterOptimizationResults.XAtMinObjective(1,2));
KernelScale3=table2array(SVM_Optimize3.HyperparameterOptimizationResults.XAtMinObjective(1,3));

t3=templateSVM('BoxConstraint',BoxCon3,'KernelScale',KernelScale3);


%Fourth optimization (used for PCA/SFS data)
SVM_Optimize4=fitcecoc(PCA_SFS_Train,TrainLabels,'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
BoxCon4=table2array(SVM_Optimize4.HyperparameterOptimizationResults.XAtMinObjective(1,2));
KernelScale4=table2array(SVM_Optimize4.HyperparameterOptimizationResults.XAtMinObjective(1,3));

t4=templateSVM('BoxConstraint',BoxCon4,'KernelScale',KernelScale4);


%Training classifiers
SVM_ULDA_mRMR_model=fitcecoc(ULDA_mRMR_Train,TrainLabels,'Learners',t1);   %Creates an SVM model
SVM_ULDA_SFS_model=fitcecoc(ULDA_SFS_Train,TrainLabels,'Learners',t2);   %Creates an SVM model
SVM_PCA_mRMR_Model=fitcecoc(PCA_mRMR_Train,TrainLabels,'Learners',t3);   %Creates an SVM model
SVM_PCA_SFS_Model=fitcecoc(PCA_SFS_Train,TrainLabels,'Learners',t4);   %Creates an SVM model


ACC_SVM_ULDA_mRMR=1-kfoldLoss(crossval(SVM_ULDA_mRMR_model));   %Determines the accuracy of the SVM model
ACC_SVM_ULDA_SFS=1-kfoldLoss(crossval(SVM_ULDA_SFS_model));   %Determines the accuracy of the SVM model
ACC_SVM_PCA_mRMR=1-kfoldLoss(crossval(SVM_PCA_mRMR_Model));   %Determines the accuracy of the SVM model
ACC_SVM_PCA_SFS=1-kfoldLoss(crossval(SVM_PCA_SFS_Model));   %Determines the accuracy of the SVM model



%% ---------------------<Export Matlab Workspace>----------------------
%Saves the trained classifiers to the workspace
save('Classifiers_new.mat','NB_ULDA_mRMR_Model','LDA_ULDA_mRMR_Model','QDA_ULDA_mRMR_model','kNN_ULDA_mRMR_model','DT_ULDA_mRMR_model','SVM_ULDA_mRMR_model', 'ULDA_Weight');


%% --------------<Plotting 10-Fold Cross Validation Results>--------------
%The accuracies of the various classifiers with the four different
%pre-processing approaches are plotted

ClassifierTypes={'NB','LDA','QDA','kNN','DT','SVM'};    %Names ofthe classifiers

%Creates vectors with the accuracies of the different classifiers with the
%various pre-processing approaches
ACCVEC_ULDA_mRMR=[ACC_NB_ULDA_mRMR,ACC_LDA_ULDA_mRMR,ACC_QDA_ULDA_mRMR,...
    ACC_kNN_ULDA_mRMR,ACC_DT_ULDA_mRMR,ACC_SVM_ULDA_mRMR];
ACCVEC_ULDA_SFS=[ACC_NB_ULDA_SFS,ACC_LDA_ULDA_SFS,ACC_QDA_ULDA_SFS,...
    ACC_kNN_ULDA_SFS,ACC_DT_ULDA_SFS,ACC_SVM_ULDA_SFS];
ACCVEC_PCA_mRMR=[ACC_NB_PCA_mRMR,ACC_LDA_PCA_mRMR,ACC_QDA_PCA_mRMR,...
    ACC_kNN_PCA_mRMR,ACC_DT_PCA_mRMR,ACC_SVM_PCA_mRMR];
ACCVEC_PCA_SFS=[ACC_NB_PCA_SFS,ACC_LDA_PCA_SFS,ACC_QDA_PCA_SFS,...
    ACC_kNN_PCA_SFS,ACC_DT_PCA_SFS,ACC_SVM_PCA_SFS];




%Plotting results

figure; %Init figure

%Plotes the accuracies
plot([1:6],ACCVEC_ULDA_mRMR,'r--o',[1:6],ACCVEC_ULDA_SFS,'b--o',[1:6],ACCVEC_PCA_mRMR,'k--o',...
    [1:6],ACCVEC_PCA_SFS,'m--o');
%Sets the tick names (classifier types)
set(gca,'xtick',[1:6],'xticklabel',ClassifierTypes);
ylim([0.5 1]);

%Title 
title('Accuracy of Six Classifiers For Four Pre-Processing Approaches');

legend('ULDA mRMR','ULDA SFS','PCA mRMR','PCA SFS');



%% Performance curve as we vary the percentage of features that we are keeping from mRMR
% (We only show this performance cuve with QDA model)

PercentFeatures=[1:2:100];  %We loop for 1 to 100% every 2%
Acc_Curve=zeros(1,length(PercentFeatures)); %Vector containing the accuracies for percentage of feature to keep
for(j=[1:length(PercentFeatures)])
    
    %--------------------<mRMR Algorithm>------------------------
    [idx,score]=fscmrmr(FeatureArray,labels);   %Index of features depending on their order of relavance
    Features_mrmr=FeatureArray(:,idx);  %Features ordered depending on relavance
    idxScores = score(idx); %index of scores depending on relevance

    % Find number of features to include
    featurePerc = PercentFeatures(j);   % Percentage of feature score to retain
    currentScore = 0;
    totalScore = sum(score);    % Total score of all features
    for i = 1:length(score)
        currentScore = currentScore + idxScores(i);
        if currentScore/totalScore > featurePerc/100    %Loops until the cumulative score of ordered features exceeds 89%
            numFeats = i;   % Number of features to use after feature selection
            break;
        end
    end

    mRMR_Features = Features_mrmr(:, 1:numFeats); % Features after feature selection
    
    %------------------------<ULDA Projection>----------------------------
    PercGoal=95;    %95 percent of total variance explained by projected data
    %mRMR Reduced Features
    [ULDA_mRMR,explained,ProjDatUnCleaned,~]=ULDA(mRMR_Features,labels,PercGoal);
    
    %------------------------<Splitting Data>-----------------------------
    cv=cvpartition(length(ULDA_mRMR(:,1)),'HoldOut',0.33);  %Train: 67%, Test: 33%
    index=cv.test;

    %Train Data
    ULDA_mRMR_Train_New=ULDA_mRMR(~index,:);    %Data for training
    TrainLabels_New=labels(~index);
    %Training LDA
    LDA_Model=fitcdiscr(ULDA_mRMR_Train_New,TrainLabels_New,'discrimtype','quadratic');   %Creates a linear model


    % ACC_LDA_B=1-kfoldLoss(LDA_B_CVModel);   %Determines average accuracy of the lda model
    Acc_Curve(j)=1-kfoldLoss(crossval(LDA_Model));   %Determines average accuracy of the lda model
end

%-------------------------<Plotting results>--------------------------

figure;
plot(PercentFeatures,Acc_Curve,'b');

%Title of plot
title('Accuracy of QDA CLassifier as Percentage Of Kept Features Increases in mRMR');



%% -----------------<Feature Selection With mRMR Results>------------------
figure;
%The names of the features
X=categorical(mRMR_Feature_Names);
X=reordercats(X,mRMR_Feature_Names);

bar(X,mRMR_Features_Percentage);   %Plotting bar chat of features with associated percentages

%Titles
ylabel('(%)');
xlabel('Feature');
title('Percentage of mRMR Score For Each Feature');
%% -----------------------<ANOVA On Test Data>----------------------------

%The best performance is only with ULDA, so we perform ULDA on the test
%data and then split this data into 10 partitions to check the accuracy of
%each classifier and then perform an ANOVA

%-------<Partition Test Data into 10 Parts and Evaluate Accuracy>----------
numParts = 10;
NB_Test_ACC=TestAcc(ULDA_mRMR_Test,TestLabels,NB_ULDA_mRMR_Model, numParts);
LDA_Test_ACC=TestAcc(ULDA_mRMR_Test,TestLabels,LDA_ULDA_mRMR_Model, numParts);
QDA_Test_ACC=TestAcc(ULDA_mRMR_Test,TestLabels,QDA_ULDA_mRMR_Model, numParts);
kNN_Test_ACC=TestAcc(ULDA_mRMR_Test,TestLabels,kNN_ULDA_mRMR_Model, numParts);
DT_Test_ACC=TestAcc(ULDA_mRMR_Test,TestLabels,DT_ULDA_mRMR_Model, numParts);
SVM_Test_ACC=TestAcc(ULDA_mRMR_Test,TestLabels,SVM_ULDA_mRMR_Model, numParts);

CNNTest = cell(augmentedTestSet.NumObservations, 1);
for i = 1:augmentedTestSet.NumObservations
    CNNTest{i} = rgb2gray(imread(augmentedTestSet.Files{i}));
end

CNNTest(151) = [];

CNN_Test_ACC = CNNTestAcc(CNNTest, CellTrain.Labels, net, numParts);

ACCArray=[NB_Test_ACC',LDA_Test_ACC',QDA_Test_ACC',kNN_Test_ACC',DT_Test_ACC',SVM_Test_ACC' CNN_Test_ACC'];

%Performing an ANOVA
[p, tbl, stats] = anova1(ACCArray);
figure
boxplot(ACCArray, 'Labels', {'NB', 'LDA', 'QDA', 'kNN', 'DT', 'SVM', 'CNN'})
title('Boxplot of Classifiers and CNN on Groups of Test Data')
ylabel('Accuracy')

%% ----------------------<Confusion Matrix>-------------------------------
load('Classifiers_new.mat');    %Loads the classifiers previously trained and saved

Result_SVM=predict(SVM_ULDA_mRMR_model,ULDA_mRMR_Test); %Results from the SVM

NewLabels=[];
PredictLabels=[];
for(i=[1:length(TestLabels)])   %Creates a vector of new label names to have stringes 
    if(TestLabels(i)==1)        %Where 1=Normal and 2=Sickle
        NewLabels{i}='Normal';
        
    else
        NewLabels{i}='Sickle';
    end
end

%Determines predicted class lables depending on the reults of the SVM
for(i=[1:length(Result_SVM)])
    if(Result_SVM(i)==1)
        PredictLabels{i}='Normal';
        
    else
        PredictLabels{i}='Sickle';
    end
end



%Plots a condusion matrix
cm=confusionchart(NewLabels,PredictLabels);
%Title of condusion matrix
cm.Title='Sickle Cell Classification Using an SVM with ULDA and mRMR';

%Shows the precision, sensitivity, and specificity
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% Finding and plotting misclassified images

%Images Classified as sickle but actually normal
Type1Error=(strcmp(NewLabels,'Normal')==true)&(strcmp(PredictLabels,'Sickle')==true);

%Images classified as normal but actually sickle
Type2Error=(strcmp(NewLabels,'Sickle')==true)&(strcmp(PredictLabels,'Normal')==true);

figure; %Init figure
%Plots a montage of the cells with type 1 error
montage(OriginalImages_Test(Type1Error));
title('Type 1 Errors');

figure; %Init figure
%Plots a montage of the cells with type 2 error
montage(OriginalImages_Test(Type2Error));
title('Type 2 Errors');
