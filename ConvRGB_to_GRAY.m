function ImGray=ConvRGB_to_GRAY(ImColour)
%Converts each RGB image in the ImColour cell array to a grayscale image
%cell array
n=length(ImColour);
ImGray=cell(1,n);   %Initializes the Grayscale image vector
for(i=[1:n])
    ImGray{i}=rgb2gray(ImColour{i});  
    
end
end