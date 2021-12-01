function cellImage = newCellImage(image, indices)
% Create new image that contains one cell.
offset = 5;    % Offset so cropped image shows entire cell
% width = 79;
[r, c] = find(indices); % Find the indices of the cell

% Find the x and y limits
xmin = min(c) - offset;
xmax = max(c) + offset;
ymin = min(r) - offset;
ymax = max(r) + offset;

wmax = max(xmax - xmin, ymax - ymin);

% Create rectangle object
rect = images.spatialref.Rectangle([xmin xmin + wmax], [ymin ymin + wmax]);

% Crop image into individual cell image
cellImage = imcrop(image, rect);



end

