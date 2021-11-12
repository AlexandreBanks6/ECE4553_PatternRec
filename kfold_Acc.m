function [acc_vec,acc_avg]=kfold_Acc(Mdl,data,labels,k,n)
%Function which trains an LDA classifier on the data 'data', with labales
%'labels' using k-fold cross validation and n=number of data samples

%Setting Parameters
acc_vec=zeros(1,k); %Vector with accuracy for each fold

for(i=[2:k])
    c=cvpartition(n,'KFold',i);
    train_ind=training(c,i);
    test_ind=test(c,i);
    train_dat=data(train_ind,:); %Train data
    test_dat=data(test_ind,:); %Test data
    train_label=labels(train_ind);
    test_label=labels(test_ind);
    %Setting random seed
    rng('default');
    tallrng('default');
    
%     Mdl=fitcdiscr(train_dat,train_label,'discrimtype','linear',...
%     'HyperparameterOptimizationOptions',...
%     struct('AcquisitionFunctionName','expected-improvement-plus')); %Training the LDA
    Mdl=fitcdiscr(train_dat,train_label,'discrimtype','linear');
    acc_vec(i-1)=1-loss(Mdl,test_dat,test_label); %Finds the accuracy for each k-fold
end
acc_avg=mean(acc_vec);
end