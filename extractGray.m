function Graynew=extractGray(ImGray,BW_Filled)
%{
Inputs:
BW_Filled=cell array of images where the cell has been extracted and filled
ImColour=original image of cell
Output:
ImGray=grayscale image
%}
n=length(BW_Filled);
Graynew=cell(1,n);   %Initializes the BW image vector
for(i=[1:n])
    %Multiplies the binary image (1=where cell is located) by the grayscale
    %image along each dimension of the RGB colour image
    Graynew{i}=bsxfun(@times,ImGray{i},cast(BW_Filled{i},'like',ImGray{i}));  
    
end
end