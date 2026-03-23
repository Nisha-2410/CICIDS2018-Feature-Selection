function [Best_pos, Best_score, curve] = IBSWO(X,Y,relief_weights,SearchAgents_no,Max_iter)

[~,dim] = size(X);

Positions = rand(SearchAgents_no,dim) > 0.5;
fitness = zeros(SearchAgents_no,1);

Best_score = inf;
Best_pos = zeros(1,dim);

curve = zeros(Max_iter,1);

%% Initial Fitness
for i = 1:SearchAgents_no
    
    fitness(i) = fitnessFunction(Positions(i,:),X,Y,relief_weights);
    
    if fitness(i) < Best_score
        Best_score = fitness(i);
        Best_pos = Positions(i,:);
    end
end

%% Main Loop
for t = 1:Max_iter
    
    for i = 1:SearchAgents_no
        
        r1 = rand; r2 = rand;
        
        % Improved exploration
        new_pos = Positions(i,:) + r1*(Best_pos - abs(Positions(i,:))) ...
                                + r2*(rand(1,dim)-0.5);
        
        % 🔥 Add Relief bias
        new_pos = new_pos + 0.1*relief_weights;
        
        % Sigmoid transfer
        S = 1 ./ (1 + exp(-2*new_pos));
        new_bin = rand(1,dim) < S;
        
        % Avoid empty subset
        if sum(new_bin)==0
            new_bin(randi(dim)) = 1;
        end
        
        % Adaptive mutation
        mutation_rate = 0.2*(1 - t/Max_iter);
        mutation_mask = rand(1,dim) < mutation_rate;
        new_bin(mutation_mask) = ~new_bin(mutation_mask);
        
        new_fit = fitnessFunction(new_bin,X,Y,relief_weights);
        
        if new_fit < fitness(i)
            Positions(i,:) = new_bin;
            fitness(i) = new_fit;
        end
        
        if new_fit < Best_score
            Best_score = new_fit;
            Best_pos = new_bin;
        end
        
        % Diversity injection
        if rand < 0.05
            Positions(i,:) = rand(1,dim) > 0.5;
        end
        
    end
    
    % Elitism
    Positions(1,:) = Best_pos;
    fitness(1) = Best_score;
    
    %% Crossover
    for k = 1:SearchAgents_no
        
        p1 = Positions(randi(SearchAgents_no),:);
        p2 = Positions(randi(SearchAgents_no),:);
        
        alpha = rand;
        child = alpha*p1 + (1-alpha)*p2;
        child = rand(1,dim) < child;
        
        if sum(child)==0
            child(randi(dim)) = 1;
        end
        
        child_fit = fitnessFunction(child,X,Y,relief_weights);
        
        [~,worst] = max(fitness);
        
        if child_fit < fitness(worst)
            Positions(worst,:) = child;
            fitness(worst) = child_fit;
        end
        
        if child_fit < Best_score
            Best_score = child_fit;
            Best_pos = child;
        end
        
    end
    
    curve(t) = Best_score;
    
    fprintf('Iteration %d Best Fitness = %f\n',t,Best_score)
    
end

end