clc;
clear;
close all;

folder = 'C:\CICIDS2018';

files = dir(fullfile(folder,'*.csv'));

disp(['Number of files found: ', num2str(length(files))]);

data = table();

for i = 1:length(files)

    filename = fullfile(folder, files(i).name);

    fprintf("Reading file %d: %s\n", i, files(i).name);

    temp = readtable(filename,'PreserveVariableNames',true);

    if isempty(data)

        data = temp;

    else

        % Find common columns between tables
        commonVars = intersect(data.Properties.VariableNames, temp.Properties.VariableNames);

        % Keep only common columns
        data = data(:,commonVars);
        temp = temp(:,commonVars);

        % Merge tables
        data = [data; temp];

    end

end

fprintf("\nTotal Samples: %d\n", height(data));
fprintf("Total Features: %d\n", width(data));

% Save merged dataset
save('merged_dataset.mat','data','-v7.3');

disp('Merged dataset saved successfully.');

