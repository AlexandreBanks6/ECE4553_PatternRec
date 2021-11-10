function BW_vec=binarizeimage(image)
BW_vec=cell(1,length(image));   %Initializes the BW image vector
for(i=[1:length(image)])
    BW=imbinarize(image{i});  %Computes binary image
    BW_vec{i}=BW(:,:,1);  %Extracts binary image
    
end
end