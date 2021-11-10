%{
Title: PipeLine1
Author: Alexandre Banks, Bailey Christian Morrel
Description: Extracts features from individual blood cell images, uses features selection,
and dimensionality reduction, and trains/evaluates classifiers
(NaiveBays,LDA,QDA,kNN,parzen,dicision trees, and SVM). There are three
class labels: circular,elongated (sickle cell),other 
Date:November 9th, 2021
Affiliation: University of New Brunswick, ECE 4553

%}

clear 
clc

%% Reading Data

circularcell_files=dir('circular\*.jpg');  %dir gives a list of all the jpg files in the database of circular cells
elongatedcell_files=dir('elongated\*.jpg');  %dir gives a list of all the jpg files in the database of circular cells
othercell_files=dir('other\*.jpg');  %dir gives a list of all the jpg files in the database of circular cells

circ_im=readimagefiles(circularcell_files,'circular'); %Circular blood cell images
elong_im=readimagefiles(elongatedcell_files,'elongated'); %Circular blood cell images
other_im=readimagefiles(othercell_files,'other'); %Circular blood cell images

%Creating labels (1==circular, 2==elongated, 3==other)
labels=[ones(1,length(circ_im)),2*ones(1,length(elong_im)),3*ones(1,length(other_im))];
%Combining datasets
ImData=[circ_im,elong_im,other_im];



%% Feature Extraction 
%(We might want to look into normalizing by number of pixels)!!!
%----------------------<Binarize each image>----------------
%Returns cell array containing all the binary images
BW_Dat=binarizeimage(ImData);   
%---------------------<Only Cells Extracted>--------------------
%Complements images, fills holes, and selects the central cell
BW_Dat_Ref_Filled=onlycell(BW_Dat);

%Complements images, and selects the central cell
BW_Dat_Ref=ComplementAndExtract(BW_Dat);
%---------------<Setting Parameters>-----------------------------
nImages=length(BW_Dat_Ref_Filled);   %Finds the number of images that we are analyzing


%Computing Perimeter
PerimVec=zeros(1,nImages);
for(i=[1:nImages])
        %NumPix=findpixnum(BW_Dat_Ref_Filled{i});           %We normalize the perimeter by the total number of pixels
        perim=bwperim(BW_Dat_Ref_Filled{i});              %Determines perimeter elements (1==perimeter)
        PerimVec(i)=sum(perim(:)==1); %Finds the number of pixels in the perimeter
end

%Computing Area of Cells
AreaVec=zeros(1,nImages);
for(i=[1:nImages])
        %NumPix=findpixnum(BW_Dat_Ref_Filled{i});           %We normalize the area by the total number of pixels
        AreaVec(i)=bwarea(BW_Dat_Ref_Filled{i}); %Approximate number of pixels in the cell
end

%Finding Circularity
CircularityVec=4*pi.*(AreaVec./(PerimVec.^2)); %Circularity calculation

%Computing Minimum Diameter
MinDiamVec=zeros(1,nImages);
for(i=[1:nImages])
        %NumPix=findpixnum(BW_Dat_Ref_Filled{i});           %We normalize the min diameter by the total number of pixels
        out=bwferet(BW_Dat_Ref_Filled{i},'MinFeretProperties'); %Finds the minimum properties of the image
        MinDiamVec(i)=out.MinDiameter(1); %Approximate minimum diameter of cells
end

%Computing Maximum Diameter
MaxDiamVec=zeros(1,nImages);
for(i=[1:nImages])
        %NumPix=findpixnum(BW_Dat_Ref_Filled{i});           %We normalize the max diameter by the total number of pixels
        out=bwferet(BW_Dat_Ref_Filled{i},'MaxFeretProperties'); %Finds the maximum properties of the image
        MaxDiamVec(i)=out.MaxDiameter(1); %Approximate maximum diameter of cells
end

%Calculating Elipticity
ElipVec=zeros(1,nImages);
for(i=[1:nImages])
        ElipVec(i)=elipticityfun(MinDiamVec(i),MaxDiamVec(i)); %Approximate maximum diameter of cells
end

%Finding Convex Hull Area
ConvexHullVec=zeros(1,nImages);
for(i=[1:nImages])
        Val=regionprops(BW_Dat_Ref{i},'ConvexArea');
        ConvexHullVec(i)=Val.ConvexArea;
end

%Computes the eccentricity of the ellipse where the values is between 0 and
%1 (0=circle, 1=line)
EccentricityVec=zeros(1,nImages);
for(i=[1:nImages])
    Val=regionprops(BW_Dat_Ref{i},'Eccentricity');    
    EccentricityVec(i)=Val.Eccentricity;
end

%Computes the equivalent diameter of the images, where the equivalent
%diameter is the diameter of a circle with the same are aas the region in
%the image
EquivDiamVec=zeros(1,nImages);
for(i=[1:nImages])
        Val=regionprops(BW_Dat_Ref{i},'EquivDiameter');
        EquivDiamVec(i)=Val.EquivDiameter;
end



%Computes solidity as the cell area/convex area
SolidityVec=zeros(1,nImages);
for(i=[1:nImages])
        Val=regionprops(BW_Dat_Refi},'Solidity');
        SolidityVec(i)=Val.Solidity;
end


%Array of features
FeatureArray=[PerimVec',AreaVec',CircularityVec',MinDiamVec',MaxDiamVec',ElipVec',ConvexHullVec',...
    EccentricityVec',EquivDiamVec',SolidityVec'];



