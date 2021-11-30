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
fun=@(XT,yT,Xt,yt) (sum(~strcmp(yt,classify(Xt,XT,yT,'linear')))); 
c=cvpartition(labels,'k',10); %10-fold cross-validation
[result,history]=sequentialfs(fun,FeatureArray,labels,'cv',c);%Perform sequential feature selection

Feature_Selected=FeatureArray(:,result);
SelectedFeatureName_Ordered=[]; %Initializes array of feature names that will be ordered

SFS_Features=zeros(length(FeatureArray(:,1)),sum(result==1)); %Vector which contains the features ordered
indexvec1=zeros(1,length(FeatureArray(1,:)));
indexvec2=zeros(1,length(FeatureArray(1,:)));
histresult=history.In;
%ordered from most to least relevant of the selected feature
for(i=[1:sum(result==1)])
    indexvec2=histresult(i,:);
    SelectedFeatureName_Ordered=[SelectedFeatureName_Ordered,FeatNames(find((indexvec2-indexvec1)==1))];
    SFS_Features(:,i)=FeatureArray(:,find((indexvec2-indexvec1)==1));
    indexvec1=indexvec2;    
end

%% Feature Ranking Using MRMR Algorithm

[idx,score]=fscmrmr(FeatureArray,labels);

Features_mrmr=FeatureArray(:,idx);
idxScores = score(idx);

% Plot of scores
figure()
bar(idxScores)

% Find number of features to include
featurePerc = 89;   % Percentage of feature score to retain
currentScore = 0;
totalScore = sum(score);    % Total score of all features
for i = 1:length(score)
    currentScore = currentScore + idxScores(i);
    if currentScore/totalScore > featurePerc/100
        numFeats = i;   % Number of features to use after feature selection
        break;
    end
end

mRMR_Features = Features_mrmr(:, 1:numFeats); % Features after feature selection
mRMR_Feature_Names=FeatNames(idx(1:numFeats));
mRMR_Features_Percentage=100*idxScores(1:numFeats)/totalScore;%(score(idx(1:numFeats))/totalScore)*100;


%% Using ULDA For Dimensionality Reduction

PercGoal=95;    %95 percent of total variance explained by projected data
%mRMR Reduced Features
[ULDA_mRMR,explained,ProjDatUnCleaned,ULDA_Weight]=ULDA(mRMR_Features,labels,PercGoal);

%SFS Reduced Features
[ULDA_SFS,FS_explained,FS_ProjDatUnCleaned, ~]=ULDA(SFS_Features,labels,PercGoal);


%% PCA For Dimensionality Reduction
%mRMR Dataset
[Coeff_mRMR_PCA,Proj_mRMR_PCA,latent_mRMR_PCA,~,explained_mRMR_PCA,Mu1]=pca(mRMR_Features,'Algorithm','eig'); 
%Coeff1=principal component coefficients, score1=representation of data in principal component domainexplained1=percentage of total variance explained
ind1=VarExpInd(explained_mRMR_PCA,95); %Returns index of last principal componet needed to explain up to 95% of the total variance
PCA_mRMR=Proj_mRMR_PCA(:,[1:ind1]); %The PCA Projection



%SFS Dataset
[Coeff_SFS_PCA,Proj_SFS_PCA,latent_SFS_PCA,~,explained_SFS_PCA,Mu2]=pca(SFS_Features,'Algorithm','eig'); 
%Coeff1=principal component coefficients, score1=representation of data in principal component domainexplained1=percentage of total variance explained
ind2=VarExpInd(explained_SFS_PCA,95); %Returns index of last principal componet needed to explain up to 95% of the total variance
PCA_SFS=Proj_mRMR_PCA(:,[1:ind2]); %The PCA Projection




%% 
%Splitting the data into training and test sets. Training sets will be used to train the model
%and perform 10-fold cross-validation. The test sets will be spit again into 10 further test sets which will be 
%used in an ANOVA to compare the accuracy between the classifiers
cv=cvpartition(length(PCA_mRMR(:,1)),'HoldOut',0.33);  %Train: 67%, Test: 33%
index=cv.test;

%Test Data
ULDA_mRMR_Test=ULDA_mRMR(index,:);
ULDA_SFS_Test=ULDA_SFS(index,:);
PCA_mRMR_Test=PCA_mRMR(index,:);
PCA_SFS_Test=PCA_SFS(index,:);

%Training Data
ULDA_mRMR_Train=ULDA_mRMR(~index,:);
ULDA_SFS_Train=ULDA_SFS(~index,:);
PCA_mRMR_Train=PCA_mRMR(~index,:);
PCA_SFS_Train=PCA_SFS(~index,:);


TestLabels=labels(index);   %Test labels
TrainLabels=labels(~index);  %Labels for training

%% Compare distribution of data between FS and ULDA

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
figure()
scatter(1:length(ULDA_mRMR), ULDA_mRMR, [], rgb)
grid on
xlabel('Observation Number')
ylabel('LDA Feature 1')
title('Scatter Plot of ULDA Feature Space with mRMR')
legend('Elongated')



%% ----------------------<Training Classifiers>------------------------
rng('default'); %Sets the random number generator for repeatability
tallrng('default');
%% ---------------------------<Naive Bayes>-----------------------------------
% ULDA_mRMR_Train
% ULDA_SFS_Train
% PCA_mRMR_Train
% PCA_SFS_Train

NB_ULDA_mRMR_Model=fitcnb(ULDA_mRMR_Train,TrainLabels);   %Creates a linear model
NB_ULDA_SFS_Model=fitcnb(ULDA_SFS_Train,TrainLabels);   %Creates a linear model
NB_PCA_mRMR_Model=fitcnb(PCA_mRMR_Train,TrainLabels);   %Creates a linear model
NB_PCA_SFS_Model=fitcnb(PCA_SFS_Train,TrainLabels);   %Creates a linear model

ACC_NB_ULDA_mRMR=1-kfoldLoss(crossval(NB_ULDA_mRMR_Model));   %Determines average accuracy of the lda model
ACC_NB_ULDA_SFS=1-kfoldLoss(crossval(NB_ULDA_SFS_Model));   %Determines average accuracy of the lda model
ACC_NB_PCA_mRMR=1-kfoldLoss(crossval(NB_PCA_mRMR_Model));   %Determines average accuracy of the lda model
ACC_NB_PCA_SFS=1-kfoldLoss(crossval(NB_PCA_SFS_Model));   %Determines average accuracy of the lda model



%% -----------------------------<LDA>-----------------------------------

% LDA_B_Model=fitcdiscr(FeatureArray,labels,'discrimtype','linear');   %Creates a linear model
LDA_ULDA_mRMR_Model=fitcdiscr(ULDA_mRMR_Train,TrainLabels,'discrimtype','linear');   %Creates a linear model
LDA_ULDA_SFS_Model=fitcdiscr(ULDA_SFS_Train,TrainLabels,'discrimtype','linear');   %Creates a linear model
LDA_PCA_mRMR_Model=fitcdiscr(PCA_mRMR_Train,TrainLabels,'discrimtype','linear');   %Creates a linear model
LDA_PCA_SFS_Model=fitcdiscr(PCA_SFS_Train,TrainLabels,'discrimtype','linear');   %Creates a linear model


% ACC_LDA_B=1-kfoldLoss(LDA_B_CVModel);   %Determines average accuracy of the lda model
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


ACC_QDA_ULDA_mRMR=1-kfoldLoss(crossval(QDA_ULDA_mRMR_model));   %Determines the accuracy of the lda model
ACC_QDA_ULDA_SFS=1-kfoldLoss(crossval(QDA_ULDA_SFS_model));   %Determines the accuracy of the lda model
ACC_QDA_PCA_mRMR=1-kfoldLoss(crossval(QDA_PCA_mRMR_Model));   %Determines the accuracy of the lda model
ACC_QDA_PCA_SFS=1-kfoldLoss(crossval(QDA_PCA_SFS_Model));   %Determines the accuracy of the lda model


%% -----------------------------<kNN>----------------------------------
%Finds the hyperparameters (k and distance measure) that minimuze the loss by using the
%automatic hyperparameter optimization and 10-fold cross validation


c=cvpartition(length(ULDA_mRMR_Train(:,1)),'Kfold',10);
% c_B=cvpartition(length(FeatureArray(:,1)),'Kfold',10);


%First Optimization
kNN_Optimize=fitcknn(ULDA_mRMR_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
k1=kNN_Optimize.NumNeighbors; %Optimal number of neighbours
DistMeas1=kNN_Optimize.Distance;

%Second Optimization
kNN_Optimize=fitcknn(ULDA_SFS_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
k2=kNN_Optimize.NumNeighbors; %Optimal number of neighbours
DistMeas2=kNN_Optimize.Distance;

%Third Optimization
kNN_Optimize=fitcknn(PCA_mRMR_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
k3=kNN_Optimize.NumNeighbors; %Optimal number of neighbours
DistMeas3=kNN_Optimize.Distance;

%Fourth Optimization
kNN_Optimize=fitcknn(PCA_SFS_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
k4=kNN_Optimize.NumNeighbors; %Optimal number of neighbours
DistMeas4=kNN_Optimize.Distance;



%Train the kNN

kNN_ULDA_mRMR_model=fitcknn(ULDA_mRMR_Train,TrainLabels,'NumNeighbors',k1,'Distance',DistMeas1);   %Creates a quadratic model
kNN_ULDA_SFS_model=fitcknn(ULDA_SFS_Train,TrainLabels,'NumNeighbors',k2,'Distance',DistMeas2);   %Creates a quadratic model
kNN_PCA_mRMR_Model=fitcknn(PCA_mRMR_Train,TrainLabels,'NumNeighbors',k3,'Distance',DistMeas3);   %Creates a quadratic model
kNN_PCA_SFS_Model=fitcknn(PCA_SFS_Train,TrainLabels,'NumNeighbors',k4,'Distance',DistMeas4);   %Creates a quadratic model


ACC_kNN_ULDA_mRMR=1-kfoldLoss(crossval(kNN_ULDA_mRMR_model));   %Determines the accuracy of the lda model
ACC_kNN_ULDA_SFS=1-kfoldLoss(crossval(kNN_ULDA_SFS_model));   %Determines the accuracy of the lda model
ACC_kNN_PCA_mRMR=1-kfoldLoss(crossval(kNN_PCA_mRMR_Model));   %Determines the accuracy of the lda model
ACC_kNN_PCA_SFS=1-kfoldLoss(crossval(kNN_PCA_SFS_Model));   %Determines the accuracy of the lda model



%% ---------------------------<Decision Tree>-------------------------
%Automatically optimizes to find the minimum leaf size hyperparameter using
%10 fold cross validation

%First optimization
DT_Optimize=fitctree(ULDA_mRMR_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
MinLeaf1=DT_Optimize.ModelParameters.MinLeaf;    %Minimum size of leaves


%Second optimization
DT_Optimize=fitctree(ULDA_SFS_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
MinLeaf2=DT_Optimize.ModelParameters.MinLeaf;    %Minimum size of leaves


%Third optimization
DT_Optimize=fitctree(PCA_mRMR_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
MinLeaf3=DT_Optimize.ModelParameters.MinLeaf;    %Minimum size of leaves


%Fourth optimization
DT_Optimize=fitctree(PCA_SFS_Train,TrainLabels,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
MinLeaf4=DT_Optimize.ModelParameters.MinLeaf;    %Minimum size of leaves



DT_ULDA_mRMR_model=fitctree(ULDA_mRMR_Train,TrainLabels,'MinLeafSize',MinLeaf1);   %Creates a quadratic model
DT_ULDA_SFS_model=fitctree(ULDA_SFS_Train,TrainLabels,'MinLeafSize',MinLeaf2);   %Creates a quadratic model
DT_PCA_mRMR_Model=fitctree(PCA_mRMR_Train,TrainLabels,'MinLeafSize',MinLeaf3);   %Creates a quadratic model
DT_PCA_SFS_Model=fitctree(PCA_SFS_Train,TrainLabels,'MinLeafSize',MinLeaf4);   %Creates a quadratic model


ACC_DT_ULDA_mRMR=1-kfoldLoss(crossval(DT_ULDA_mRMR_model));   %Determines the accuracy of the lda model
ACC_DT_ULDA_SFS=1-kfoldLoss(crossval(DT_ULDA_SFS_model));   %Determines the accuracy of the lda model
ACC_DT_PCA_mRMR=1-kfoldLoss(crossval(DT_PCA_mRMR_Model));   %Determines the accuracy of the lda model
ACC_DT_PCA_SFS=1-kfoldLoss(crossval(DT_PCA_SFS_Model));   %Determines the accuracy of the lda model



%% ------------------------------<SVM>--------------------------------
%Trains a multiclass error-correcting outputs codes using k(k-1)/2 binary
%support vector machine. We also standardize the predictors
%Optimize the hyperparameters
SVM_Optimize1=fitcecoc(ULDA_mRMR_Train,TrainLabels,'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
% SVM_B_Optimize=fitcecoc(FeatureArray,labels,'OptimizeHyperparameters','auto',...
% 'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c_B,...
%     'ShowPlots',false,'Verbose',0));
BoxCon1=table2array(SVM_Optimize1.HyperparameterOptimizationResults.XAtMinObjective(1,2));
% BoxCon_B=table2array(SVM_B_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,2));
KernelScale1=table2array(SVM_Optimize1.HyperparameterOptimizationResults.XAtMinObjective(1,3));
% KernelScale_B=table2array(SVM_B_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,3));

t1=templateSVM('BoxConstraint',BoxCon1,'KernelScale',KernelScale1);
% t_B=templateSVM('BoxConstraint',BoxCon_B,'KernelScale',KernelScale_B);



SVM_Optimize2=fitcecoc(ULDA_SFS_Train,TrainLabels,'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
% SVM_B_Optimize=fitcecoc(FeatureArray,labels,'OptimizeHyperparameters','auto',...
% 'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c_B,...
%     'ShowPlots',false,'Verbose',0));
BoxCon2=table2array(SVM_Optimize2.HyperparameterOptimizationResults.XAtMinObjective(1,2));
% BoxCon_B=table2array(SVM_B_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,2));
KernelScale2=table2array(SVM_Optimize2.HyperparameterOptimizationResults.XAtMinObjective(1,3));
% KernelScale_B=table2array(SVM_B_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,3));

t2=templateSVM('BoxConstraint',BoxCon2,'KernelScale',KernelScale2);
% t_B=templateSVM('BoxConstraint',BoxCon_B,'KernelScale',KernelScale_B);



SVM_Optimize3=fitcecoc(PCA_mRMR_Train,TrainLabels,'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
% SVM_B_Optimize=fitcecoc(FeatureArray,labels,'OptimizeHyperparameters','auto',...
% 'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c_B,...
%     'ShowPlots',false,'Verbose',0));
BoxCon3=table2array(SVM_Optimize3.HyperparameterOptimizationResults.XAtMinObjective(1,2));
% BoxCon_B=table2array(SVM_B_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,2));
KernelScale3=table2array(SVM_Optimize3.HyperparameterOptimizationResults.XAtMinObjective(1,3));
% KernelScale_B=table2array(SVM_B_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,3));

t3=templateSVM('BoxConstraint',BoxCon3,'KernelScale',KernelScale3);
% t_B=templateSVM('BoxConstraint',BoxCon_B,'KernelScale',KernelScale_B);



SVM_Optimize4=fitcecoc(PCA_SFS_Train,TrainLabels,'OptimizeHyperparameters','auto',...
'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c,...
    'ShowPlots',false,'Verbose',0));
% SVM_B_Optimize=fitcecoc(FeatureArray,labels,'OptimizeHyperparameters','auto',...
% 'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','CVPartition',c_B,...
%     'ShowPlots',false,'Verbose',0));
BoxCon4=table2array(SVM_Optimize4.HyperparameterOptimizationResults.XAtMinObjective(1,2));
% BoxCon_B=table2array(SVM_B_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,2));
KernelScale4=table2array(SVM_Optimize4.HyperparameterOptimizationResults.XAtMinObjective(1,3));
% KernelScale_B=table2array(SVM_B_Optimize.HyperparameterOptimizationResults.XAtMinObjective(1,3));

t4=templateSVM('BoxConstraint',BoxCon4,'KernelScale',KernelScale4);
% t_B=templateSVM('BoxConstraint',BoxCon_B,'KernelScale',KernelScale_B);



SVM_ULDA_mRMR_model=fitcecoc(ULDA_mRMR_Train,TrainLabels,'Learners',t1);   %Creates a quadratic model
SVM_ULDA_SFS_model=fitcecoc(ULDA_SFS_Train,TrainLabels,'Learners',t2);   %Creates a quadratic model
SVM_PCA_mRMR_Model=fitcecoc(PCA_mRMR_Train,TrainLabels,'Learners',t3);   %Creates a quadratic model
SVM_PCA_SFS_Model=fitcecoc(PCA_SFS_Train,TrainLabels,'Learners',t4);   %Creates a quadratic model


ACC_SVM_ULDA_mRMR=1-kfoldLoss(crossval(SVM_ULDA_mRMR_model));   %Determines the accuracy of the lda model
ACC_SVM_ULDA_SFS=1-kfoldLoss(crossval(SVM_ULDA_SFS_model));   %Determines the accuracy of the lda model
ACC_SVM_PCA_mRMR=1-kfoldLoss(crossval(SVM_PCA_mRMR_Model));   %Determines the accuracy of the lda model
ACC_SVM_PCA_SFS=1-kfoldLoss(crossval(SVM_PCA_SFS_Model));   %Determines the accuracy of the lda model



%% ---------------------<Export Matlab Workspace>----------------------
save('Classifiers_new.mat','NB_ULDA_mRMR_Model','LDA_ULDA_mRMR_Model','QDA_ULDA_mRMR_model','kNN_ULDA_mRMR_model','DT_ULDA_mRMR_model','SVM_ULDA_mRMR_model', 'ULDA_Weight');


%% --------------<Plotting 10-Fold Cross Validation Results>--------------
% BarLabels=categorical({'NB','LDA','QDA','kNN','DT','SVM'});  %Labels for Bar Graph
% 
% Acc=[ACC_NB,ACC_FS_NB,ACC_FS_ULDA_NB;ACC_LDA,ACC_FS_LDA,ACC_FS_LDA_ULDA;...
%     ACC_QDA,ACC_FS_QDA,ACC_FS_ULDA_QDA;ACC_kNN,ACC_FS_kNN,ACC_FS_ULDA_kNN;...
%     ACC_DT,ACC_FS_DT,ACC_FS_ULDA_DT;ACC_SVM,ACC_FS_SVM,ACC_FS_ULDA_SVM];
% figure;
% bar(BarLabels,Acc);
% legend('ULDA','MRMR','MRMR and ULDA');
% xlabel('Classifier');
% ylabel('Average Accuracy');
% ylim([0.5 1.1]);
% title('Accuracy of 6 Classifiers With 3 Pre-Processing Approaches Using 10-fold Cross Validation');
ClassifierTypes={'NB','LDA','QDA','kNN','DT','SVM'};
ACCVEC_ULDA_mRMR=[ACC_NB_ULDA_mRMR,ACC_LDA_ULDA_mRMR,ACC_QDA_ULDA_mRMR,...
    ACC_kNN_ULDA_mRMR,ACC_DT_ULDA_mRMR,ACC_SVM_ULDA_mRMR];
ACCVEC_ULDA_SFS=[ACC_NB_ULDA_SFS,ACC_LDA_ULDA_SFS,ACC_QDA_ULDA_SFS,...
    ACC_kNN_ULDA_SFS,ACC_DT_ULDA_SFS,ACC_SVM_ULDA_SFS];
ACCVEC_PCA_mRMR=[ACC_NB_PCA_mRMR,ACC_LDA_PCA_mRMR,ACC_QDA_PCA_mRMR,...
    ACC_kNN_PCA_mRMR,ACC_DT_PCA_mRMR,ACC_SVM_PCA_mRMR];
ACCVEC_PCA_SFS=[ACC_NB_PCA_SFS,ACC_LDA_PCA_SFS,ACC_QDA_PCA_SFS,...
    ACC_kNN_PCA_SFS,ACC_DT_PCA_SFS,ACC_SVM_PCA_SFS];




%Plotting results

figure;

plot([1:6],ACCVEC_ULDA_mRMR,'r--o',[1:6],ACCVEC_ULDA_SFS,'b--o',[1:6],ACCVEC_PCA_mRMR,'k--o',...
    [1:6],ACCVEC_PCA_SFS,'m--o');
set(gca,'xtick',[1:6],'xticklabel',ClassifierTypes);
ylim([0.5 1]);
title('Accuracy of Six Classifiers For Four Pre-Processing Approaches');

legend('ULDA mRMR','ULDA SFS','PCA mRMR','PCA SFS');



%% Performance curve as we vary the percentage of features that we are keeping
% (We only show this performance cuve with xx model)

PercentFeatures=[1:2:100];
Acc_Curve=zeros(1,length(PercentFeatures)); %Vector containing the accuracies for percentage of feature to keep
for(j=[1:length(PercentFeatures)])
    
    %--------------------<mRMR Algorithm>------------------------
    [idx,score]=fscmrmr(FeatureArray,labels);
    Features_mrmr=FeatureArray(:,idx);
    idxScores = score(idx);

    % Find number of features to include
    featurePerc = PercentFeatures(j);   % Percentage of feature score to retain
    currentScore = 0;
    totalScore = sum(score);    % Total score of all features
    for i = 1:length(score)
        currentScore = currentScore + idxScores(i);
        if currentScore/totalScore > featurePerc/100
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
    ULDA_mRMR_Train_New=ULDA_mRMR(~index,:);
    TrainLabels_New=labels(~index);
    %Training LDA
    LDA_Model=fitcdiscr(ULDA_mRMR_Train_New,TrainLabels_New,'discrimtype','quadratic');   %Creates a linear model


    % ACC_LDA_B=1-kfoldLoss(LDA_B_CVModel);   %Determines average accuracy of the lda model
    Acc_Curve(j)=1-kfoldLoss(crossval(LDA_Model));   %Determines average accuracy of the lda model
end

%-------------------------<Plotting results>--------------------------

figure;
plot(PercentFeatures,Acc_Curve,'b');
title('Accuracy of QDA CLassifier as Percentage Of Kept Features Increases in mRMR');



%% -----------------<Feature Selection With mRMR Results>------------------
figure;
X=categorical(mRMR_Feature_Names);
X=reordercats(X,mRMR_Feature_Names);
bar(X,mRMR_Features_Percentage);   %Plotting bar chat of features with associated percentages
ylabel('(%)');
xlabel('Feature');
title('Percentage of mRMR Score For Each Feature');
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
