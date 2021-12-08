function ImGray=ConvRGB_to_GRAY(ImColour)
%{
Title: ConvRGB_to_Gray
Authors: Alexandre Banks, Christian Morrell
Date:November 9th, 2021
Affiliation: University of New Brunswick, ECE 4553
Description: Converts each RGB image in the ImColour cell array to a grayscale image
%cell array
%}

n=length(ImColour); %Length of colour image array
ImGray=cell(1,n);   %Initializes the Grayscale image vector
for(i=[1:n])    %Loops for the number of images in a cell array
    ImGray{i}=rgb2gray(ImColour{i});    %Converts individual images to gray
    
end
end