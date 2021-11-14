function cellImage = newCellImage(image, indices)
% Create new image that contains one cell.
offset = 25;
[r, c] = find(indices);
% xmin = min(c) - offset;    % Minimum x value
% ymin = min(r) - offset;    % Minimum y value
% width = max(c) - xmin + offset;  % Width of rectangle
% height = max(r) - ymin + offset; % Height of rectangle
% rect = [xmin ymin width height];
xmin = min(c) - offset;
xmax = max(c) + offset;
ymin = min(r) - offset;
ymax = max(r) + offset;
rect = images.spatialref.Rectangle([xmin xmax], [ymin ymax]);
cellImage = imcrop(image, rect);
% Filter noise using Gaussian filter
% av_filter = fspecial('average', [3 3]);
% av_g = imfilter(cellImage, av_filter, 'replicate');
% gauss_g = imgaussfilt(cellImage, 1);
% med_g = medfilt2(cellImage);
end

