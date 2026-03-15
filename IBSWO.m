function [Best_pos, Best_score, curve] = IBSWO(X, Y, SearchAgents_no, Max_iter)

% Number of Features
[~, dim] = size(X);

% Lower and Upper bounds
lb = 0;
ub = 1;

% Initialize Population
Positions = rand(SearchAgents_no, dim) > 0.5;

Best_pos = zeros(1, dim);
Best_score = inf;
curve = zeros(1, Max_iter);

for i = 1:SearchAgents_no
    fitness = fitnessFunction(Positions(i,:), X, Y);
    
    if fitness < Best_score
        Best_score = fitness;
        Best_pos = Positions(i,:);
    end
end

% Main Loop
for t = 1:Max_iter
    
    for i = 1:SearchAgents_no
        
        r1 = rand();
        r2 = rand();
        
        % Exploration Phase
        if rand < 0.5
            new_pos = Positions(i,:) + r1*(Best_pos - Positions(i,:));
        else
            new_pos = Positions(i,:) + r2*(Positions(randi(SearchAgents_no),:) - Positions(i,:));
        end
        
        % Sigmoid Transfer Function
        S = 1 ./ (1 + exp(-new_pos));
        new_bin = rand(1,dim) < S;
        
        % Fitness Evaluation
        new_fit = fitnessFunction(new_bin, X, Y);
        
        old_fit = fitnessFunction(Positions(i,:), X, Y);
        
        if new_fit < old_fit
            Positions(i,:) = new_bin;
        end
        
        % Update Global Best
        if new_fit < Best_score
            Best_score = new_fit;
            Best_pos = new_bin;
        end
        
    end
    
    curve(t) = Best_score;
    
    fprintf('Iteration %d Best Fitness = %f\n', t, Best_score);
    
end

end