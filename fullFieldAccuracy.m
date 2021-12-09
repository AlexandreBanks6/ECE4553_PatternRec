function accuracy = fullFieldAccuracy(predictions, labels)
% Get accuracy of full field images
difference = labels - predictions;  % difference between labels and predictions
idx = find(difference);
numWrong = length(idx);
total = length(predictions);

% Get accuracy of predictions
accuracy = (total - numWrong)/total;
end

