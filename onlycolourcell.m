function ImExtract=onlycolourcell(ImColour,BW_Filled)
%{
Inputs:
BW_Filled=cell array of images where the cell has been extracted and filled
ImColour=original image of cell
Output:
ImExtract=extracted colour image where only the cell is present
%}
n=length(BW_Filled);
ImExtract=cell(1,n);   %Initializes the BW image vector
for(i=[1:n])
    %Multiplies the binary image (1=where cell is located) by the colour
    %image along each dimension of the RGB colour image
    ImExtract{i}=bsxfun(@times,ImColour{i},cast(BW_Filled{i},'like',ImColour{i}));  
    
end
end