function fitness = fitnessFunction(selected_features, X, Y)
% Inputs:
%   selected_features - binary vector from IBSWO (1=selected, 0=not)
%   X                 - feature matrix [n x f]
%   Y                 - class labels [n x 1]
% Output:
%   fitness           - scalar (lower = better)

    % --- Weights ---
    alpha = 0.99;  % classification error weight
    beta  = 0.01;  % feature count penalty weight

    total_features = size(X, 2);

    % --- No features selected edge case ---
    if sum(selected_features) == 0
        fitness = 1;
        return;
    end

    % --- Apply binary mask ---
    X_selected = X(:, logical(selected_features));

    % --- KNN with 10-fold Cross Validation ---
    k = 5;
    cv = cvpartition(Y, 'KFold', 10, 'Stratify', true);

    correct = 0;
    total   = 0;

    for fold = 1:cv.NumTestSets

        X_train = X_selected(cv.training(fold), :);
        Y_train = Y(cv.training(fold));
        X_test  = X_selected(cv.test(fold),     :);
        Y_test  = Y(cv.test(fold));

        mdl    = fitcknn(X_train, Y_train, 'NumNeighbors', k, ...
                         'Distance', 'euclidean');
        Y_pred = predict(mdl, X_test);

        correct = correct + sum(Y_pred == Y_test);
        total   = total   + numel(Y_test);

    end

    % --- Fitness ---
    error_rate    = 1 - (correct / total);
    feature_ratio = sum(selected_features) / total_features;
    fitness       = alpha * error_rate + beta * feature_ratio;

end
