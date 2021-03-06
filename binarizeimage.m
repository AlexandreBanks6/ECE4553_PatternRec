function BW_vec=binarizeimage(image)
%{
Title: binarizeimage
Author: Alexandre Banks, Christian  Morrel
Date:November 14th, 2021
Affiliation: University of New Brunswick, ECE 4553
Desription:
Creates a cell array containing binary images
Input: image (grayscale image)
Output: BW_vec (binary cell array)
%}
BW_vec=cell(1,length(image));   %Initializes the BW image vector
for(i=[1:length(image)])
    BW_vec{i}=imbinarize(image{i});  %Computes binary image
    
end
end