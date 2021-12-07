function accuracy = fullFieldAccuracy(predictions, labels)
difference = labels - predictions;
idx = find(difference);
numWrong = length(idx);
total = length(predictions);
accuracy = (total - numWrong)/total;
end

