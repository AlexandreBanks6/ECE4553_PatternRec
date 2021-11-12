function [BestAccuracy,BestModel]=BestAcc(CVMdl,data,labels,k)

%Setting Parameters
acc_vec=zeros(1,k); %Vector with accuracy for each fold
for(i=[1:k])
    acc_vec(i)=1-loss(CVMdl.Trained{i},data,labels);
end
BestAccuracy=max(acc_vec);
BestModel=CVMdl.Trained{find(acc_vec==BestAccuracy)}; %Finds model with best accuracy
end