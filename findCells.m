function [cells, numCells] = findCells(image)
%%% Find cell clusters within a given image using the layer matrix.

minimumArea = 3000; % Minimum area to determine if object is a cell
cellIdx = cell(100, 1);    % Preallocate cell index array

tempImage = imbinarize(image, 'adaptive', 'Sensitivity', 1);    % binarize image with high sensitivity threshold

cellArray=imfill(~tempImage,'holes');    %Fills the holes

CC = bwconncomp(cellArray);    % Find all connected components in image   

% Count number of cells
numCells = 0;   
for i = 1:CC.NumObjects
    idx = CC.PixelIdxList{i};
    area = length(idx);
    if area > minimumArea
        numCells = numCells + 1;
        cellIdx{numCells} = idx;    % Save cell index
%         falseArr = false(size(image));   % Default logical array
%         falseArr(idx) = 1;
%         cells{k} = falseArr;
%         k = k + 1;
    end
end

% Create array of individual cells
cells = cell(numCells, 1);  % Preallocate array

for i = 1:numCells
    falseArr = false(size(image));   % Default logical array
    idx = cellIdx{i};
    falseArr(idx) = 1;
    cells{i} = falseArr;
end

% for j = 1:length(cells)
% imshowpair(cells{j}, image, 'montage')
% end


end

