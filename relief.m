function [weights, selected_idx] = relief(X, y, m, k)

[n, f] = size(X);

weights = zeros(1,f);

% Feature range (for normalization)
range_f = max(X) - min(X) + eps;

for i = 1:m
    
    % Random sample
    r = randi(n);
    Ri = X(r,:);
    yi = y(r);
    
    % Compute distances (vectorized)
    dist = sum((X - Ri).^2, 2);
    
    % Sort neighbors
    [~, idx_sorted] = sort(dist);
    
    % Remove itself
    idx_sorted(idx_sorted == r) = [];
    
    % Find hits and misses
    hit_idx = idx_sorted(y(idx_sorted) == yi);
    miss_idx = idx_sorted(y(idx_sorted) ~= yi);
    
    % Take k nearest
    hit_idx = hit_idx(1:min(k, length(hit_idx)));
    miss_idx = miss_idx(1:min(k, length(miss_idx)));
    
    % Compute updates
    for h = 1:length(hit_idx)
        hit = X(hit_idx(h),:);
        weights = weights - abs(Ri - hit) ./ range_f;
    end
    
    for m_i = 1:length(miss_idx)
        miss = X(miss_idx(m_i),:);
        weights = weights + abs(Ri - miss) ./ range_f;
    end
    
end

% Normalize
weights = weights / m;

% Rank features
[~, ranked_idx] = sort(weights, 'descend');

% Keep top features
num_keep = round(0.6 * f);
selected_idx = ranked_idx(1:num_keep);

end