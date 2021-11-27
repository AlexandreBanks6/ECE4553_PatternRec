%% Read In Data

circularcell_files=dir('circular\*.jpg');  %dir gives a list of all the jpg files in the database of circular cells
elongatedcell_files=dir('elongated\*.jpg');  %dir gives a list of all the jpg files in the database of circular cells
othercell_files=dir('other\*.jpg');  %dir gives a list of all the jpg files in the database of circular cells

circ_im=readimagefiles(circularcell_files,'circular'); %Circular blood cell images
elong_im=readimagefiles(elongatedcell_files,'elongated'); %Circular blood cell images
other_im=readimagefiles(othercell_files,'other'); %Circular blood cell images


%% Converting to Gray Scale

%Converting circular images to grayscale
for(i=[1:length(circ_im)])
    GrayIm=rgb2gray(circ_im{i});
    imwrite(GrayIm,['GrayImages/other/',num2str(i),'.jpg']); 
end

%Converting other images to grayscale
for(j=[i+1:i+length(other_im)])
    GrayIm=rgb2gray(other_im{j-i});
    imwrite(GrayIm,['GrayImages/other/',num2str(j),'.jpg']); 
end

%Converting elongated images to grayscale
for(i=[1:length(elong_im)])
    GrayIm=rgb2gray(other_im{i});
    imwrite(GrayIm,['GrayImages/elongated/',num2str(i),'.jpg']); 
end

