function fitness = fitnessFunction(position, X, Y, relief_weights)

% Avoid empty subset
if sum(position) == 0
    fitness = 1;
    return
end

X_selected = X(:, logical(position));

%% Normalize Relief weights
w = (relief_weights - min(relief_weights)) ./ ...
    (max(relief_weights) - min(relief_weights) + eps);

%% HoldOut validation (fast)
cv = cvpartition(Y,'HoldOut',0.3);

Xtrain = X_selected(training(cv),:);
Ytrain = Y(training(cv));

Xtest = X_selected(test(cv),:);
Ytest = Y(test(cv));

%% AdaBoost classifier
t = templateTree('MaxNumSplits',20);

mdl = fitcensemble(Xtrain,Ytrain,...
    'Method','AdaBoostM1',...
    'NumLearningCycles',20,...
    'Learners',t);

pred = predict(mdl,Xtest);

%% Accuracy (MAIN OBJECTIVE)
accuracy = mean(pred == Ytest);
loss = 1 - accuracy;

%% Feature ratio (VERY LIGHT penalty)
feature_ratio = sum(position)/length(position);

%% Relief guidance (STRONGER now)
importance = mean(w(logical(position)));

%% 🔥 NEW WEIGHTS (RELAXED)
alpha = 0.97;   % focus on accuracy
beta  = 0.005;  % very small penalty
gamma = 0.025;  % encourage good features

%% Final fitness
fitness = alpha*loss + beta*feature_ratio - gamma*importance;

end