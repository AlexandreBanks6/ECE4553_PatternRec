function cells = findCells(image, layers)
%%% Find cell clusters within a given image using the layer matrix.

cells = cell(20, 1);    % Assume >= 20 cells in each image

% Search through and find where element is equal to 2 (cell)
indices = find(layers == 2);
[r, c] = find(layers == 2);  % Elements that are in the cell layer
previousCells = false(size(image));

% Check to make sure that this cell has not been checked already:
% AND the current cell with all previous ones, if any elements are > 0 then
% cell already has been counted
k = 1;
for i = 1:length(r)
    % THIS IS WAY TOO SLOW
    if r(i) ~= 0
        newCell = grayconnected(image, r(i), c(i));
        if ~any(newCell & previousCells, 'all')
            % Cell does not already exist
            previousCells = previousCells | newCell;   % Save new cell into previous cells so it doesn't get counted twice
            cells{k} = newCell; % Save new cell in array
            k = k + 1;
%             newIdx = find(newCell);
%             otherIdx = find(indices == newIdx);
%             r(otherIdx) = 0;
%             c(otherIdx) = 0;
            
            %         [cellr, cellc] = find(newCell);
            %         r = r(r~=cellr);
            %         c = c(c~=cellc);
        end
    end
end

% newCellIndices = grayconnected(grayImage, 111, 1908);   % Get one specific cluster

end

