function [ theta_k, theta_all, i ] = loss_SGD( X, C, theta, batch_size, max_epoch )
    
    [sample_size, n_samples] = size(X);
    [n_classes, ~] = size(C);
    
    [theta_size, ~] = size(theta);
    
    theta_all = zeros(theta_size, max_epoch);
    theta_k = theta;
    
    alpha_base = batch_size / n_samples;
    
    for i = 1:max_epoch
        idxs = randperm(n_samples);
        for k = 1:(n_samples / batch_size)
            idx_k = idxs((k-1) * batch_size + 1 : k * batch_size);
            X_k = X(:,idx_k);
            C_k = C(:,idx_k);
            
            b_k = theta_k(1:n_classes);
            W_k = reshape(theta_k(n_classes+1:theta_size), [sample_size,n_classes]);
            
            g_k = loss_grad_theta(X_k, C_k, W_k, b_k); % todo: fix zeros to b vector

            if i <= 100
                a_k = alpha_base /(10);
            else
                a_k = alpha_base /(sqrt(i));
            end
            theta_k = theta_k - a_k * g_k;
        end
        theta_all(:, i) = vertcat(b_k,W_k(:));
    end

end