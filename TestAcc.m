function [ACC_Vec]=TestAcc(TestData,TestLabels,Model)
TestLength=floor(length(TestData(:,1))/20);    %Split test data into 20 sections
ACC_Vec=zeros(1,20);
for(i=[1:20])
    Dat=TestData([(i-1)*TestLength+1:i*TestLength]);
    Labels=TestLabels([(i-1)*TestLength+1:i*TestLength]);
    ACC_Vec(i)=1-loss(Model,Dat,Labels);
end
end