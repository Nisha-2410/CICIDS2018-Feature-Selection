function [weights, selected_idx] = relief(X, y, m)

[n, f] = size(X);

weights = zeros(1,f);

for i = 1:m
    
    % Random Instance
    r = randi(n);
    Ri = X(r,:);
    yi = y(r);
    
    % Distance from all points
    dist = sum((X - Ri).^2,2);
    
    % Nearest Hit and Miss
    hit_dist = inf;
    miss_dist = inf;
    
    for j = 1:n
        
        if j == r
            continue;
        end
        
        if y(j) == yi
            if dist(j) < hit_dist
                hit = X(j,:);
                hit_dist = dist(j);
            end
        else
            if dist(j) < miss_dist
                miss = X(j,:);
                miss_dist = dist(j);
            end
        end
        
    end
    
    % Weight Update
    for k = 1:f
        
        weights(k) = weights(k) ...
            + (Ri(k)-miss(k))^2 ...
            - (Ri(k)-hit(k))^2;
        
    end
    
end

% Normalize weights
weights = weights / m;

% Select features (Paper threshold)
[~, ranked_idx] = sort(weights,'descend');

num_keep = round(0.6 * f);   % keep top 60%

selected_idx = ranked_idx(1:num_keep);


end
