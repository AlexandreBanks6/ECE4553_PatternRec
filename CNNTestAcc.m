function ACC_Vec = CNNTestAcc(TestData,TestLabels,net, numParts)
TestLength=floor(length(TestData(:,1))/numParts);    %Split test data into sections
ACC_Vec=zeros(1,numParts);
for(i=[1:numParts])
    Dat=cell2table(TestData([(i-1)*TestLength+1:i*TestLength]));
    
    
    Labels=TestLabels([(i-1)*TestLength+1:i*TestLength]);
    YPred = classify(net, Dat);
    ACC_Vec(i) = mean(YPred == Labels);
end
end

