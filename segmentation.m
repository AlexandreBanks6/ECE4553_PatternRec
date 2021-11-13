%%% Image Segmentation
%%% Christian Morrell, Alexandre Banks
%%% November 11, 2021

%%% Goal: We want to take a blood slide image and produce an individual
%%% image for each cell within the full field image

close all
clear
clc

%% Load in full field images

% Load raw data
numImages = 50; % Number of images in database 2
fullFiles = cell(numImages, 1); % Preallocate array
fullImages = cell(numImages, 1);    % Preallocate array
imageName = '\source.jpg';   % Name of source images in dataset

for i = 1:numImages
    
    if i < 10
        % Need to append 0 to find proper file folder
        dirName = ['erythrocytesIDB2\0' num2str(i) 'erythrocytesIDB2'];
    else
        dirName = ['erythrocytesIDB2\' num2str(i) 'erythrocytesIDB2'];
    end
    
    fullFiles{i} = dir([dirName imageName]);    % Get jpg file
    temp  = readimagefiles(fullFiles{i}, dirName);  % Load image
    fullImages{i} = temp{1}; 
    
end


% Convert RGB images to grayscale
grayImages = ConvRGB_to_GRAY(fullImages)';


%% Split up full field images into datasets

% rgbImage = imread('source.jpg');
% grayImage = imadjust(imadjust(rgb2gray(rgbImage))); % Convert rgb image to grayscale, increase contrast
% figure()
% imshow(grayImage);
% for i = 1:numImages
%     grayImage = imadjust(grayImages{i});    % Contrast cells from background
%     
%     % Get individual cell
%     layers = imsegkmeans(grayImage, 2); % Segment image into two layers
%     cells = findCells(layers);    % Find cells in image
%     for j = 1:length(cells)
%         newCellIndices = grayconnected(grayImage, cells(j, 1), cells(j, 2));
%     end
%     
% end

% binImage = imbinarize(grayImage);  %% Convert to binary image
% imshow(binImage)

grayImage = imadjust(grayImages{1});

% Get individual cell
layers = imsegkmeans(grayImage, 2);  % Segment image into two layers
cells = findCells(grayImage, layers);
% newCellIndices = grayconnected(grayImage, 111, 1908);   % Get one specific cluster
% imshow(labeloverlay(grayImage, newCellIndices))

% Crop image to show one cell
cellImage = newCellImage(grayImage, newCellIndices);
figure()
imshow(cellImage)


%% BAD/OLD CODE

% boundaries = bwboundaries(BW);  % coordinates of all boundaries in binary image
% figure()
% imshow(BW)
% hold on
% for k = 1:length(boundaries)
%     boundary = boundaries{k};
%     plot(boundary(:, 2), boundary(:, 1), 'r', 'LineWidth', 2)
% end

% B = labeloverlay(grayImage, L);
% imshow(B)


% test = boundaries{125};
% xmin = min(test(:, 1));
% xmax = max(test(:, 1));
% ymin = min(test(:, 2));
% ymax = max(test(:, 2));
% width = xmax - xmin;
% height = ymax - ymin;
% rect = [xmin ymin width height];
%
% I2 = imcrop(BW, rect);
% figure()
% imshow(I2)

% edgeI = edge(BW);
% imshow(edgeI)

% hey = BW(test(:, 1), test(:, 2));
% figure()
% imshow(hey)