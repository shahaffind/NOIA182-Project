function [ Theta_k, Theta_all, i ] = ResNN_SGD( X, C, Theta, n_layers, batch_size, max_epoch, Yv, Cv )
    
    [~, n_samples] = size(X);
    
%     Theta_size = size(Theta);
    
%     Theta_all = zeros(Theta_size(1), max_epoch);
    Theta_all = 0;
    Theta_k = Theta;

    alpha_base = batch_size / (n_samples);
    
%     for i = 1:max_epoch
    i = 0;
    while 1
        i = i+1;
        idxs = randperm(n_samples);
        for k = 1:(n_samples / batch_size)
            idx_k = idxs((k-1) * batch_size + 1 : k * batch_size);
            X_k = X(:,idx_k);
            C_k = C(:,idx_k);
            
            [~, ~, Xis] = forward_pass(X_k, C_k, Theta_k, n_layers);
            g_k = back_propagation(Xis, C_k, Theta_k, n_layers);
            
            if i <= 100
                a_k = alpha_base / 10;
            else
                a_k = alpha_base /(sqrt(i));
            end
            Theta_k = Theta_k - a_k * g_k;
        end
%         Theta_all(:, i) = Theta_k;
        
        if mod(i, 10) == 0
            [all_proba, loss_val, ~] = forward_pass(Yv, Cv, Theta_k, n_layers);
            [c_p, R] = correct_percent(all_proba, Cv);
            viewFeatures2D(Yv, R);
            drawnow update
            fprintf('iter:%d\n\tloss: %f\n\tcorrect: %f\n\n', i, loss_val, c_p);
        end
    end
end

