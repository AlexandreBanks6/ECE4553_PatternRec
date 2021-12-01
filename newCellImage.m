function cellImage = newCellImage(image, indices)
% Create new image that contains one cell.
offset = 25;    % Offset so cropped image shows entire cell
width = 299;
[r, c] = find(indices); % Find the indices of the cell

% Find the x and y limits
xmin = min(c) - offset;
% xmax = max(c) + offset;
ymin = min(r) - offset;
% ymax = max(r) + offset;

% Create rectangle object
rect = images.spatialref.Rectangle([xmin xmin + width], [ymin ymin + width]);

% Crop image into individual cell image
cellImage = imcrop(image, rect);



% Filter noise using Gaussian filter
% av_filter = fspecial('average', [3 3]);
% av_g = imfilter(cellImage, av_filter, 'replicate');
% gauss_g = imgaussfilt(cellImage, 1);
% med_g = medfilt2(cellImage);
end

