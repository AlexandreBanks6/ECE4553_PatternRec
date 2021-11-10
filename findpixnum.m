function numPix=findpixnum(BW)
%Finds the total number of pixels in the image to normalize the data by
[row,col]=size(BW);
numPix=row*col;
end