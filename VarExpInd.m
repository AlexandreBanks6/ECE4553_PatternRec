function [ind]=VarExpInd(explained,GoalPerc)
    %Returns the number of principal components that are required to explain up
    %to 95% of the variance of the data

    Perc=0; %Stores percent of total variance explained
    ind=1; %number of principal components explaining up to 95% of variance

    for(i=[1:length(explained)])
        Perc=explained(i)+Perc; %Adds percentage of total variance explained
        if(Perc>=GoalPerc)
            ind=i;
            break
        end
    end
end