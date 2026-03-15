clc;
clear;
close all;

disp('Loading merged dataset...')

load('merged_dataset.mat');   % contains variable "data"

fprintf("Original Samples: %d\n", height(data));
fprintf("Original Features: %d\n", width(data));

%% -----------------------------
% 1 Remove Duplicate Rows
%% -----------------------------

disp("Removing duplicate rows...")

data = unique(data);

fprintf("Samples after duplicate removal: %d\n", height(data));


%% -----------------------------
% 2 Handle Infinite Values
%% -----------------------------

disp("Handling Inf values...")

X = data(:,1:end-1);

X = table2array(X);

X(isinf(X)) = NaN;


%% -----------------------------
% 3 Handle Missing Values
%% -----------------------------

disp("Handling missing values...")

X = fillmissing(X,'median');


%% -----------------------------
% 4 Separate Labels
%% -----------------------------

disp("Extracting labels...")

labels = data{:,end};


%% -----------------------------
% 5 Encode Labels
%% -----------------------------

disp("Encoding labels...")

[unique_labels,~,label_encoded] = unique(labels);

Y = label_encoded;

fprintf("Number of Classes: %d\n", length(unique_labels));


%% -----------------------------
% 6 Remove Zero Variance Features
%% -----------------------------

disp("Removing zero variance features...")

variance = var(X);

X(:,variance==0) = [];

fprintf("Remaining Features: %d\n", size(X,2));


%% -----------------------------
% 7 Normalization (Min-Max)
%% -----------------------------

disp("Normalizing features...")

X = (X - min(X)) ./ (max(X) - min(X) + eps);


%% -----------------------------
% 8 SMART SAMPLING (IMPORTANT)
%% -----------------------------
% CICIDS2018 has millions of rows
% We keep balanced samples from each class

disp("Applying smart sampling...")

samples_per_class = 5000;   % adjustable

X_sampled = [];
Y_sampled = [];

classes = unique(Y);

for i = 1:length(classes)

    idx = find(Y == classes(i));

    if length(idx) > samples_per_class
        idx = idx(randperm(length(idx), samples_per_class));
    end

    X_sampled = [X_sampled; X(idx,:)];
    Y_sampled = [Y_sampled; Y(idx)];

end

fprintf("Sampled Dataset Size: %d samples\n", length(Y_sampled));


%% -----------------------------
% 9 Train Test Split
%% -----------------------------

disp("Splitting dataset...")

cv = cvpartition(Y_sampled,'HoldOut',0.3);

Xtrain = X_sampled(training(cv),:);
Ytrain = Y_sampled(training(cv));

Xtest = X_sampled(test(cv),:);
Ytest = Y_sampled(test(cv));

fprintf("Training Samples: %d\n", length(Ytrain));
fprintf("Testing Samples: %d\n", length(Ytest));


%% -----------------------------
% 10 Save Processed Dataset
%% -----------------------------

save('processed_dataset.mat','Xtrain','Ytrain','Xtest','Ytest','unique_labels','-v7.3')

disp("Preprocessing completed successfully.")