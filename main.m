clc;
clear;
close all;

[X_train, X_test, y_train, y_test] = preprocess('train_test_network.csv');

% ============================
% 2️⃣ Apply Manual ReliefF
% ============================

m = 1000;   % no. of iterations

[weights, selected_idx] = relief(X_train, y_train, m);

X_train_relief = X_train(:, selected_idx);
X_test_relief  = X_test(:, selected_idx);

fprintf("Original Features: %d\n", size(X_train,2));
fprintf("After ReliefF: %d\n", size(X_train_relief,2));

% ============================
% 3️⃣ IBSWO Optimization
% ============================

SearchAgents_no = 25;
Max_iter = 40;

disp('Running IBSWO Optimization...');

[Best_pos, Best_score, curve] = IBSWO(X_train_relief, y_train, SearchAgents_no, Max_iter);

save('bestSolution.mat','Best_pos');

% ============================
% 4️⃣ Load Saved IBSWO Result
% ============================

load('bestSolution.mat');

X_train_opt = X_train_relief(:, logical(Best_pos));
X_test_opt  = X_test_relief(:, logical(Best_pos));

fprintf('Selected Features after IBSWO: %d\n', size(X_train_opt,2));

% ============================
% SVM Classification (Multi-Class)
% ============================

t = templateSVM('KernelFunction','rbf','Standardize',true);

svmModel = fitcecoc(X_train_opt, y_train,'Learners',t);

svmPred = predict(svmModel, X_test_opt);

svmAcc = sum(svmPred == y_test) / length(y_test);

fprintf('\nSVM Accuracy: %.4f\n', svmAcc);

% ============================
% Decision Tree Classification
% ============================

dtModel = fitctree(X_train_opt, y_train);

dtPred = predict(dtModel, X_test_opt);

dtAcc = sum(dtPred == y_test) / length(y_test);

fprintf('Decision Tree Accuracy: %.4f\n', dtAcc);

% ============================
% Random Forest Classification
% ============================

rfModel = TreeBagger(100, X_train_opt, y_train, ...
    'Method','classification');

rfPred = predict(rfModel, X_test_opt);

rfPred = str2double(rfPred);   % Convert cell to numeric

rfAcc = sum(rfPred == y_test) / length(y_test);

fprintf('Random Forest Accuracy: %.4f\n', rfAcc);



