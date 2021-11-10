function [BW_Refined]=selectcell(BW_Image)
% %Function takes in a binary image which can containg other objects as well
% %as the cell, and returns binary image with just the cell
% 
% %We want to extract the single largest blob in the image
% %We label the image and find the number of blobs
% [labelimage,numBlobs]=bwlabel(BW_Image);
% blobmeas=regionprops(labelimage,'area');    %Extracts the area of the blobs
% AreaList=blobmeas.Area;   %Creates a list of all the area measurements
% 
% %Sort the areas of the blob in descending order
% [sortAreas,sortInd]=sort(AreaList,'descend');
% LargestBlob=ismember(labelimage,sortInd(1));
% %Converts from a labeled image intoa binary image
% BW_Refined=LargestBlob>0;



end