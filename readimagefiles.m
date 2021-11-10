function images=readimagefiles(imagefiles,folder)
%Returns cell array of images from the images specified by the list in
%"imagefiles"
nfiles=length(imagefiles);
images=cell(1,nfiles);
for(i=[1:nfiles])   %Loops for the number of images in the folder
    currfilename=imagefiles(i).name;    %Finds the image name
    images{i}=imread([folder,'\',currfilename]); %Reads in the image
end

end
