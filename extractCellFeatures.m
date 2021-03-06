function [featureSet, featureNames] = extractCellFeatures(images)

%{
Title: extractCellFeatures
Author: Alexandre Banks, Christian  Morrel
Date:November 15th, 2021
Affiliation: University of New Brunswick, ECE 4553
Desription:
Extracts14 shape and texture related features of grayscale images.

Input:
images = cell array of grayscale images
featuresSet = array of features where each columnis a feature and each row is an image
 The features used are: [Perimeter,Area,Circularity,Minimum Diameter,
Maximum Diameter,Elipticity,Convex Area,Eccentricity,Equivalent Diameter,Solidity,
Texture,Symmetry (from flipping image left-to-right),
SymmetryUpDown (from flipping image up and down),SymmetryRotate (from rotating the image by 90 degrees)]

featureNames = array of corresponding features names
%}

%----------------------<Binarize each image>----------------
%Returns cell array containing all the binary images
BW_Dat=binarizeimage(images);   
%---------------------<Only Cells Extracted>--------------------
%Complements images, fills holes, and selects the central cell
BW_Dat_Ref_Filled=onlycell(BW_Dat);

%Complements images, and selects the central cell
BW_Dat_Ref=ComplementAndExtract(BW_Dat);

%---------------<Only Grayscale Image With Cell>-------------
images=extractGray(images,BW_Dat_Ref_Filled);

%---------------<Setting Parameters>-----------------------------
nImages=length(BW_Dat_Ref_Filled);   %Finds the number of images that we are analyzing


%Computing Perimeter
PerimVec=zeros(1,nImages);
for(i=[1:nImages])  %Loops for the number of images
        perim=bwperim(BW_Dat_Ref_Filled{i});              %Determines perimeter elements (1==perimeter)
        PerimVec(i)=sum(perim(:)==1); %Finds the number of pixels in the perimeter
end

%Computing Area of Cells
AreaVec=zeros(1,nImages);
for(i=[1:nImages])
        AreaVec(i)=bwarea(BW_Dat_Ref_Filled{i}); %Approximate number of pixels in the cell
end

%Finding Circularity
CircularityVec=4*pi.*(AreaVec./(PerimVec.^2)); %Circularity calculation

%Computing Minimum Diameter
MinDiamVec=zeros(1,nImages);
for(i=[1:nImages])
        out=bwferet(BW_Dat_Ref_Filled{i},'MinFeretProperties'); %Finds the minimum properties of the image
        MinDiamVec(i)=out.MinDiameter(1); %Approximate minimum diameter of cells
end

%Computing Maximum Diameter
MaxDiamVec=zeros(1,nImages);
for(i=[1:nImages])
        out=bwferet(BW_Dat_Ref_Filled{i},'MaxFeretProperties'); %Finds the maximum properties of the image
        MaxDiamVec(i)=out.MaxDiameter(1); %Approximate maximum diameter of cells
end

%Calculating Elipticity
ElipVec=zeros(1,nImages);
for(i=[1:nImages])
        ElipVec(i)=elipticityfun(MinDiamVec(i),MaxDiamVec(i)); %Ellipticity is calculated from approximate maximum and minimum diameter
end

%Finding Convex Hull Area
ConvexHullVec=zeros(1,nImages);
for(i=[1:nImages])
        Val=regionprops(BW_Dat_Ref{i},'ConvexArea');    %Uses the regionprops function to calculate the convex area
        ConvexHullVec(i)=Val.ConvexArea;
end

%Computes the eccentricity of the ellipse where the values is between 0 and
%1 (0=circle, 1=line)
EccentricityVec=zeros(1,nImages);
for(i=[1:nImages])
    Val=regionprops(BW_Dat_Ref_Filled{i},'Eccentricity');      %regionprops used for eccentricity
    EccentricityVec(i)=Val.Eccentricity;
end

%Computes the equivalent diameter of the images, where the equivalent
%diameter is the diameter of a circle with the same are aas the region in
%the image
EquivDiamVec=zeros(1,nImages);
for(i=[1:nImages])
        Val=regionprops(BW_Dat_Ref{i},'EquivDiameter'); %regionprops() used to calculate equivalent diameter
        EquivDiamVec(i)=Val.EquivDiameter;
end



%Computes solidity as the cell area/convex area
SolidityVec=zeros(1,nImages);
for(i=[1:nImages])
        Val=regionprops(BW_Dat_Ref{i},'Solidity');  %Using regionprops() to calculate the solidity
        SolidityVec(i)=Val.Solidity;
end

%Computes the texture of the images
%(Standard Deviation of Grayscale values)
TextureVec=zeros(1,nImages);
for(i=[1:nImages])
        X=images{i}; %Grayscale Image Matrix
        TextureVec(i)=std(im2double(X),0,'all');
end

%Symmetry (Calculated as mean squared error between image and flipped
%image)
SymmetryVec=zeros(1,nImages);
SymmetryVec2=zeros(1,nImages);
SymmetryVec3=zeros(1,nImages);
for(i=[1:nImages])
        X=images{i};
        SymmetryVec(i)=immse(X,fliplr(X));    %Calculates symmetry by flipping left and right
        SymmetryVec2(i)=immse(X,flipud(X));    %Calculates symmetry by flipping up and down
        SymmetryVec3(i)=immse(X,imrotate(X,90));    %Calculates symmetry by rotating by 90 degrees 
        
end
%Array of features
featureSet=[PerimVec',AreaVec',CircularityVec',MinDiamVec',MaxDiamVec',ElipVec',ConvexHullVec',...
    EccentricityVec',EquivDiamVec',SolidityVec',TextureVec',SymmetryVec',SymmetryVec2',SymmetryVec3'];
%Array of feature names
featureNames={'Perim','Area','Circularity','MinDiam','MaxDiam','Elipticity','ConvexArea',...
    'Eccentricity','EquivalentDiam','Solidity','Texture','Symmetry','SymmetryUpDown','SymmetryRotate'};
end

