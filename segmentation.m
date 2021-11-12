%%% Image Segmentation
%%% Christian Morrell, Alexandre Banks
%%% November 11, 2021

%%% Goal: We want to take a blood slide image and produce an individual
%%% image for each cell within the full field image

close all
clear
clc

%% Load in image

rgbImage = imread('source.jpg');
grayImage = imadjust(rgb2gray(rgbImage)); % Convert rgb image to grayscale, increase contrast
imshow(grayImage);

BW = imbinarize(grayImage);  %% Convert to binary image
imshow(BW)

% Get individual cell
L = imsegkmeans(grayImage, 2);  % Segment image into two layers
J = grayconnected(grayImage, 680, 9);   % Get one specific cluster
imshow(labeloverlay(grayImage, J))

% Crop image to show one cell
offset = 100;
[r, c] = find(J);
xmin = min(r) - offset;    % Minimum x value
ymin = min(c) - offset;    % Minimum y value
width = max(r) - xmin;  % Width of rectangle
height = max(c) - ymin; % Height of rectangle
rect = [xmin ymin width height];
J2 = imcrop(grayImage, rect);
figure()
imshow(J2)


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