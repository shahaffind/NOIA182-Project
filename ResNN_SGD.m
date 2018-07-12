function [ Theta_k, records ] = ResNN_SGD( X, C, Theta, n_layers, batch_size, learning_rate, moment_gamma, max_epoch, Yv, Cv, record_every )
    
    [~, n_samples] = size(X);
    
    records = zeros(ceil(max_epoch / record_every), 2);
    Theta_k = Theta;
    m_k = zeros(size(Theta));
    tic;
    for i=1:max_epoch
        idxs = randperm(n_samples);
        for k = 1:(n_samples / batch_size)
            idx_k = idxs((k-1) * batch_size + 1 : k * batch_size);
            X_k = X(:,idx_k);
            C_k = C(:,idx_k);
            
            [~, ~, Xis] = forward_pass(X_k, C_k, Theta_k, n_layers);
            g_k = back_propagation(Xis, C_k, Theta_k, n_layers);
            
            if i <= 100
                a_k = learning_rate;
                gamma = moment_gamma;
            else
                a_k = learning_rate * 5 / sqrt(i);
                gamma = moment_gamma * 10 / sqrt(i);
            end
            
            m_proj = g_k * m_k' * g_k / (norm(g_k)*norm(g_k));
            m_rej = m_k - m_proj;
            m_k = m_rej;
            
            m_k = a_k * g_k + gamma * m_k;
            Theta_k = Theta_k - m_k;
        end
        
        if mod(i, record_every) == 0
            [all_proba, loss_val, ~] = forward_pass(Yv, Cv, Theta_k, n_layers);
            [c_p, R] = correct_percent(all_proba, Cv);
            records(i / record_every, :) = [loss_val, c_p];
            toc;
            
%             viewFeatures2D(Yv, R);
%             drawnow update
            
            fprintf('iter:%d\n\tloss: %f\n\tcorrect: %f\n\n', i, loss_val, c_p);
        end
    end
end

