function cells = findCells(image)
%%% Find cell clusters within a given image using the layer matrix.

cells = cell(20, 1);    % Assume >= 20 cells in each image

tempImage = imbinarize(image, 'adaptive', 'Sensitivity', 1);    % binarize image with high sensitivity threshold

cellArray=imfill(~tempImage,'holes');    %Fills the holes


new = bwconncomp(cellArray);    % Find all connected components in image   
def = false(size(image));   % Default logical array

k = 1;
for i = 1:new.NumObjects
    idx = new.PixelIdxList{i};
    area = length(idx);
    if area > 3000
        def(idx) = 1;
        cells{k} = def;
        k = k + 1;
    end
    
end
% for j = 1:length(cells)
% imshowpair(cells{j}, image, 'montage')
% end


end

