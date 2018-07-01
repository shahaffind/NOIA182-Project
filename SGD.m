function [ W_k, W_all, i ] = SGD( g, X, C, W, b_size, max_epoch )
    
    samples_size = size(X);
    n_samples = samples_size(2);
    
    w_size = size(W);
    
    W_all = zeros((w_size(1)+1)*w_size(2), max_epoch);
    W_k = zeros((w_size(1)+1)*w_size(2), 1);
    W_k(1:w_size(1)*w_size(2)) = W(:);
    
    for i = 1:max_epoch
        idxs = randperm(n_samples);
        for k = 1:(n_samples / b_size)
            idx_k = idxs((k-1) * b_size + 1 : k * b_size);
            X_k = X(:,idx_k);
            C_k = C(:,idx_k);
            
            W_for_grad = W_k(1:w_size(1)*w_size(2));
            W_for_grad = reshape(W_for_grad, size(W));
            
            g_k = g(X_k, C_k, W_for_grad, zeros(w_size(2), 1)); % todo: fix zeros to b vector

            if norm(g_k) < 1e-3
                break
            end

            d_k = -g_k;
            a_k = 1/k;

            W_k = W_k + a_k * d_k;
        end
        W_all(:, i) = W_k;
    end

end