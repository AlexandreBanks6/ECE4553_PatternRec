function [BW_Refined]=onlycell(BW_Image)
%{
Title: onlycell
Author: Alexandre Banks, Christian  Morrel
Date:November 14th, 2021
Affiliation: University of New Brunswick, ECE 4553
Desription: This script reads in a cell array of binary images, and then
complements the image, fills the holes and extracts the blob with the
largest area.

Input:
BW_Image (binary image)

Output:
BW_Refined = array of binary images only containing cell with the largest
area

%}
n=length(BW_Image);
BW_Refined=cell(1,n);
for(i=[1:n])
    BW_comp=imcomplement(BW_Image{i});  %complements the image where the cells=ones (light spots)
    BW_fill=imfill(BW_comp,'holes');    %Fills the holes
    BW_Refined{i}=bwareafilt(BW_fill,1);    %Extracts the blob with the largest area    
end

end