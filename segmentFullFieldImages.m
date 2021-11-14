%%% Segments dataset of full field images into individual cells.
%%% Arguments:
%%%             images: cell array of grayscale images
%%% Outputs:
%%%             segmentedDataset: cell array containing each image
%%%             segmented into individual cells

function segmentedDataset = segmentFullFieldImages(images)

numImages = length(images); % Get number of images in dataset

segmentedDataset = cell(numImages, 1);   % Array to hold dataset of full field images divided into individual cells
for i = 1:numImages
    image = imadjust(images{i});    % Contrast cells from background
    
    % Get individual cells
    [cells, numCells] = findCells(image);   % Get cells as logical arrays
    
    individualCells = cell(numCells, 1);   % Array to hold dataset from full field image
    
    for j = 1:numCells
        % Store each individual cell image in dataset
        cellImage = newCellImage(image, cells{j});  % Crop image to show one cell
        individualCells{j} = cellImage;
    end
    
    segmentedDataset{i} = individualCells;    % Store dataset in cell array of image datasets
    
end

save('Segmented.mat', 'segmentedDataset')   % Save variable
end

