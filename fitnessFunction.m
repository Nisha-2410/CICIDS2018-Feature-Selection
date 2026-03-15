function fitness = fitnessFunction(position, X, Y)

% If no feature selected
if sum(position)==0
    fitness = 1;
    return;
end

% Selected Features
X_selected = X(:, logical(position));

% Train KNN Classifier (Paper Used Wrapper Approach)
mdl = fitcknn(X_selected, Y, 'NumNeighbors',5);

% 5-Fold Cross Validation
CVmdl = crossval(mdl,'KFold',5);
loss = kfoldLoss(CVmdl);

% Feature Reduction Ratio
alpha = 0.99;
beta = 0.01;

feature_ratio = sum(position)/length(position);

% FINAL FITNESS FUNCTION (EXACT PAPER STYLE)
fitness = alpha*loss + beta*feature_ratio;

end