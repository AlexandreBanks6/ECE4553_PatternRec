function [ProjDat,explained,ProjDatUnCleaned, W]=ULDA(Data,Labels,PercGoal)
%{
Function which uses LDA for dimensionality reduction, and returns the data
projected into the new feature space and also the percentage that each
component explains. ProjDatUnCleaned represents the data projected on the
new feature space (but feature space not reduced depending on PercGoal
percentage of total variance).
%}
Mdl=fitcdiscr(Data,Labels,'DiscrimType','linear'); %Fits the LDA Model
%Calculates eigenvectors (W) and eigenvalues (LAMBDA)
[W,LAMBDA]=eig(Mdl.BetweenSigma,Mdl.Sigma);     %Using between class and within class covariance matrices
lambda=diag(LAMBDA);  %Only taking diagonal elements of eigenvalues (all off-diagonals are zero)
%Sorting the eigenvectors by decreasing eigenvalues
[lambda,SortOrder]=sort(lambda,'descend');
W=W(:,SortOrder);

%Determining the percent explained
eig_sum=sum(lambda);    %Sum of eigenvalues
explained=zeros(1,length(lambda));
for(i=[1:length(lambda)])
    explained(i)=(lambda(i)/eig_sum)*100;   %Calculates percentage that each eigenvalue explains of total eigenvalues
end
ProjDatUnCleaned=Data*W; %Returns all the projected data components without accounting for the goal percentage of overall variance
ind=VarExpInd(explained,PercGoal); %Determined indexes ofeigenvectors explaining up to 95% of variance 
W=W(:,[1:ind]); %only taking eigenvectors explaining up to PercGoal %
explained=explained([1:ind]);   %Only returning percentage of total variance of the eigenvectors explaining up to PerGoal %
ProjDat=Data*W;
end