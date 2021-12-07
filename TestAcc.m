function [ACC_Vec]=TestAcc(TestData,TestLabels,Model, numParts)
TestLength=floor(length(TestData(:,1))/numParts);    %Split test data into sections
ACC_Vec=zeros(1,numParts);
for(i=[1:numParts])
    Dat=TestData([(i-1)*TestLength+1:i*TestLength]);
    Labels=TestLabels([(i-1)*TestLength+1:i*TestLength]);
    ACC_Vec(i)=1-loss(Model,Dat,Labels);
end
end