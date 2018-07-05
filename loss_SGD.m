function [ Theta_k, records, i ] = loss_SGD( X, C, theta, batch_size, max_epoch, Yv, Cv, record_every )
    
    [sample_size, n_samples] = size(X);
    [n_classes, ~] = size(C);
    
    [theta_size, ~] = size(theta);
    
    records = zeros(ceil(max_epoch / record_every), 2);
    
    Theta_k = theta;
    
    for i = 1:max_epoch
        idxs = randperm(n_samples);
        for k = 1:(n_samples / batch_size)
            idx_k = idxs((k-1) * batch_size + 1 : k * batch_size);
            X_k = X(:,idx_k);
            C_k = C(:,idx_k);
            
            b_k = Theta_k(1:n_classes);
            W_k = reshape(Theta_k(n_classes+1:theta_size), [sample_size,n_classes]);
            
            g_k = loss_grad_theta(X_k, C_k, W_k, b_k);

            if i <= 100
                a_k = 1 / 100;
            else
                a_k = 1 / (10*sqrt(i));
            end
            Theta_k = Theta_k - a_k * g_k;
        end
        
        if mod(i, record_every) == 0
            b_k = Theta_k(1:n_classes);
            W_k = reshape(Theta_k(n_classes+1:theta_size), [sample_size,n_classes]);
            
            loss_val = loss(Yv, Cv, W_k, b_k);
            all_proba = softmax(Yv, W_k, b_k);
            [c_p, R] = correct_percent(all_proba, Cv);
            records(i / record_every, :) = [loss_val, c_p];
            
            viewFeatures2D(Yv, R);
            drawnow update
            fprintf('iter:%d\n\tloss: %f\n\tcorrect: %f\n\n', i, loss_val, c_p);
        end
    end

end