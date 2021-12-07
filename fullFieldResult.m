function result = fullFieldResult(ratio ,threshold)

if (threshold > ratio) && (ratio > 0)
    % Reject
    result = 3;
elseif ratio > threshold
    % Sickle
    result = 2;
else
    % Normal
    result = 1;
end

end

