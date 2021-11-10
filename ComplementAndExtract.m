function [BW_Refined]=ComplementAndExtract(BW_Image)
n=length(BW_Image);
BW_Refined=cell(1,n);
for(i=[1:n])
    BW_comp=imcomplement(BW_Image{i});  %complements the image where the cells=ones (light spots)
    BW_Refined{i}=bwareafilt(BW_comp,1);    %Extracts the blob with the largest area    
end

end