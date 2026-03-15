function fitness = fitnessFunction(selected_features, X, Y)
% FITNESSFUNCTION Evaluates quality of selected feature subset
% Inputs:
%   selected_features - binary vector (1 = selected, 0 = not selected)
%   X                 - feature matrix [samples x features]
%   Y                 - class labels [samples x 1]
% Output:
%   fitness           - fitness value (lower is better)

    % --- Parameters ---
    alpha = 0.99;   % weight for classification error
    beta  = 0.01;   % weight for feature count penalty

    % --- Total number of features ---
    total_features = size(X, 2);

    % --- Handle case: no features selected ---
    if sum(selected_features) == 0
        fitness = 1;   % worst possible fitness
        return;
    end

    % --- Extract selected feature columns ---
    X_selected = X(:, logical(selected_features));

    % --- Classification using K-Nearest Neighbor (KNN) ---
    k = 5;
    indices = crossvalind('Kfold', Y, 10);   % 10-fold cross-validation
    
    correct = 0;
    total   = 0;
    
    for fold = 1:10
        test_idx  = (indices == fold);
        train_idx = ~test_idx;

        X_train = X_selected(train_idx, :);
        Y_train = Y(train_idx);
        X_test  = X_selected(test_idx,  :);
        Y_test  = Y(test_idx);

        % Train and predict
        mdl        = fitcknn(X_train, Y_train, 'NumNeighbors', k);
        Y_pred     = predict(mdl, X_test);
        correct    = correct + sum(Y_pred == Y_test);
        total      = total   + numel(Y_test);
    end

    % --- Classification Error Rate ---
    error_rate = 1 - (correct / total);

    % --- Feature Ratio (penalty for using many features) ---
    feature_ratio = sum(selected_features) / total_features;

    % --- Final Fitness (lower is better) ---
    fitness = alpha * error_rate + beta * feature_ratio;

end
```

---

### How the fitness function works

**Formula:**
```
fitness = α × error_rate + β × feature_ratio
