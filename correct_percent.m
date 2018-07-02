function [ c_p, R ] = correct_percent( all_proba, C )
    
    [n_samples, n_classes] = size(all_proba);

    [~,idx_max] = max(all_proba', [], 1);
    idx_max = idx_max + (0:n_samples-1) * n_classes;
    R = zeros(n_classes, n_samples);
    R(idx_max) = 1;
    
    correct = R .* C;
    
    c_p = sum(correct(:)) / n_samples;
end
