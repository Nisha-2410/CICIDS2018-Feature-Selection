clc;
clear;
close all;

disp("Loading merged dataset...")
load('merged_dataset.mat');   % variable: data

fprintf("Original Samples: %d\n", height(data));
fprintf("Original Features: %d\n", width(data));

%% ===============================
% 1. REMOVE DUPLICATES
%% ===============================
disp("Removing duplicates...")
data = unique(data);

%% ===============================
% 2. REMOVE IRRELEVANT COLUMNS
%% ===============================
cols_to_remove = {'Timestamp','Flow ID','Src IP','Dst IP','Source IP','Destination IP'};

for i = 1:length(cols_to_remove)
    if any(strcmp(data.Properties.VariableNames, cols_to_remove{i}))
        data.(cols_to_remove{i}) = [];
    end
end

%% ===============================
% 3. EXTRACT LABELS
%% ===============================
labels = data{:,end};

if iscell(labels) || isstring(labels)
    labels = categorical(labels);
end

if iscategorical(labels)
    Y = double(labels ~= 'BENIGN');   % 0 = benign, 1 = attack
else
    Y = double(labels);
end

%% ===============================
% 4. BALANCED SAMPLING (OPTIONAL)
%% ===============================
disp("Applying controlled sampling...")

samples_per_class = 8000;  % slightly higher for better learning

benign_idx = find(Y == 0);
attack_idx = find(Y == 1);

n_benign = min(samples_per_class, length(benign_idx));
n_attack = min(samples_per_class, length(attack_idx));

benign_idx = benign_idx(randperm(length(benign_idx), n_benign));
attack_idx = attack_idx(randperm(length(attack_idx), n_attack));

idx = [benign_idx; attack_idx];

data_small = data(idx,1:end-1);
Y = Y(idx);

%% ===============================
% 5. CONVERT TO NUMERIC
%% ===============================
disp("Converting features to numeric...")

X = zeros(height(data_small), width(data_small));

for i = 1:width(data_small)

    col = data_small{:,i};

    if iscell(col) || isstring(col)
        col = str2double(col);
    elseif iscategorical(col)
        col = double(col);
    elseif islogical(col)
        col = double(col);
    elseif isdatetime(col)
        col = posixtime(col);
    end

    X(:,i) = double(col);
end

%% ===============================
% 6. HANDLE INF VALUES
%% ===============================
disp("Handling Inf values...")
X(isinf(X)) = NaN;

%% ===============================
% 7. HANDLE MISSING VALUES (MEDIAN)
%% ===============================
disp("Handling missing values using median...")

for i = 1:size(X,2)
    col = X(:,i);
    col(isnan(col)) = median(col(~isnan(col)));
    X(:,i) = col;
end

%% ===============================
% 8. REMOVE ZERO VARIANCE FEATURES
%% ===============================
disp("Removing zero variance features...")

v = var(X);
X(:,v == 0) = [];

fprintf("Remaining Features: %d\n", size(X,2));

%% ===============================
% 9. TRAIN-TEST SPLIT (NO LEAKAGE)
%% ===============================
disp("Splitting dataset...")

n = length(Y);
idx = randperm(n);

train_size = round(0.7 * n);

train_idx = idx(1:train_size);
test_idx = idx(train_size+1:end);

Xtrain = X(train_idx,:);
Ytrain = Y(train_idx);

Xtest = X(test_idx,:);
Ytest = Y(test_idx);

fprintf("Training Samples: %d\n", length(Ytrain));
fprintf("Testing Samples: %d\n", length(Ytest));

%% ===============================
% 10. NORMALIZATION (AFTER SPLIT)
%% ===============================
disp("Normalizing data (NO DATA LEAKAGE)...")

min_val = min(Xtrain);
max_val = max(Xtrain);

Xtrain = (Xtrain - min_val) ./ (max_val - min_val + eps);
Xtest  = (Xtest  - min_val) ./ (max_val - min_val + eps);

%% ===============================
% 11. SAVE DATA
%% ===============================
save('processed_dataset.mat','Xtrain','Ytrain','Xtest','Ytest','-v7.3')

disp("Preprocessing completed successfully.")