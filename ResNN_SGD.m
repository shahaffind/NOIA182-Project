function [ Theta_k, Theta_all, i ] = ResNN_SGD( X, C, Theta, n_layers, batch_size, max_epoch )
    
    Theta_k = Theta;
    
    samples_size = size(X);
    n_samples = samples_size(2);
    
    Theta_size = size(Theta);
    
    Theta_all = zeros(Theta_size(1), max_epoch);
    
    for i = 1:max_epoch
        idxs = randperm(n_samples);
        for k = 1:(n_samples / batch_size)
            idx_k = idxs((k-1) * batch_size + 1 : k * batch_size);
            X_k = X(:,idx_k);
            C_k = C(:,idx_k);
            
            [~, ~, Xis] = forward_pass(X_k, C_k, Theta_k, n_layers);
            g_k = back_propagation(Xis, C_k, Theta_k, n_layers);

            if norm(g_k) < 1e-3
                break
            end

            d_k = -g_k;
            if i <= 50
                a_k = 1/100;
            else
                a_k = 1/1000;
            end

            Theta_k = Theta_k + a_k * d_k;
        end
        Theta_all(:, i) = Theta_k;
    end

end

