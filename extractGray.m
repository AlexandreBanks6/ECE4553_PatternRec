function Graynew=extractGray(ImGray,BW_Filled)
%{
Title: extractGray
Author: Alexandre Banks, Christian  Morrel
Date:November 14th, 2021
Affiliation: University of New Brunswick, ECE 4553
Desription:
Multiplies binary image with only cell extracted with grayscale image.

Inputs:
BW_Filled=cell array of images where the cell has been extracted and filled
ImGray=grayscale image of cell
Output:
GrayNew=grayscale image with only cell extracted
%}
n=length(BW_Filled);
Graynew=cell(1,n);   %Initializes the BW image vector
for(i=[1:n])
    %Multiplies the binary image (1=where cell is located) by the grayscale
    %image along each dimension of the RGB colour image
    Graynew{i}=bsxfun(@times,ImGray{i},cast(BW_Filled{i},'like',ImGray{i}));  
    
end
end