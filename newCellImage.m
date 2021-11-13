function cellImage = newCellImage(image, indices)
% Create new image that contains one cell.
offset = 50;
[r, c] = find(indices);
xmin = min(c) - offset;    % Minimum x value
ymin = min(r) - offset;    % Minimum y value
width = max(c) - xmin + offset;  % Width of rectangle
height = max(r) - ymin + offset; % Height of rectangle
rect = [xmin ymin width height];
cellImage = imcrop(image, rect);
end

