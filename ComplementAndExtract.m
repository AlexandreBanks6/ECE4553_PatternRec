function [BW_Refined]=ComplementAndExtract(BW_Image)
%{
Title: ComplementAndExtract
Author: Alexandre Banks, Christian  Morrel
Date:November 14th, 2021
Affiliation: University of New Brunswick, ECE 4553
Desription: Complements grayscale images and extracts the blob with the
largest area (does not fill any holes)

Input:
BW_Image (array of binary images)
BW_Refined (array of extracted and complemented cell images)

%}
n=length(BW_Image);
BW_Refined=cell(1,n);
for(i=[1:n])
    BW_comp=imcomplement(BW_Image{i});  %complements the image where the cells=ones (light spots)
    BW_Refined{i}=bwareafilt(BW_comp,1);    %Extracts the blob with the largest area    
end

end