function [ g_theta ] = back_propagation( Xis, C, Theta, n_layers )
    
    [sample_size, n_samples] = size(Xis(:,:,1));
    [n_labels, ~] = size(C);
    
    theta_layer_size = sample_size + (sample_size^2) * 2; % 2* NxN matrix + N vector
    
    loss_weights_start_loc = n_layers*((sample_size^2)*2 + sample_size);
    loss_weights_end_loc = loss_weights_start_loc + n_labels +  sample_size * n_labels;
    b_loss = Theta(loss_weights_start_loc + 1 : loss_weights_start_loc + n_labels);
    W_loss = Theta(loss_weights_start_loc + 1 + n_labels : loss_weights_end_loc);
    W_loss = reshape(W_loss, [sample_size, n_labels]);
    
    g_theta = zeros(size(Theta));
    g_theta(loss_weights_start_loc + 1 : loss_weights_end_loc) = loss_grad_theta(Xis(:,:,n_layers+1), C, W_loss, b_loss);
    
    g_x = loss_grad_x(Xis(:,:,n_layers+1), C, W_loss, b_loss);
    g_x = sum(g_x, 2) / n_samples;
    
    for k = n_layers:-1:1
        layer_weights = Theta((k - 1)*theta_layer_size + 1 : k*theta_layer_size);
        b = layer_weights(1 : sample_size);
        W1 = layer_weights(sample_size + 1 : sample_size + sample_size^2);
        W2 = layer_weights(sample_size + sample_size^2 + 1 : theta_layer_size);
        W1 = reshape(W1, [sample_size,sample_size]);
        W2 = reshape(W2, [sample_size,sample_size]);
        
        g_w_layer = ResNN_jac_theta_t_mul(Xis(:,:,k), W1, W2, b, g_x);
        
        g_theta((k-1)*theta_layer_size + 1 : k*theta_layer_size) = g_w_layer;
        
        g_x = ResNN_jac_x_t_mul(Xis(:,:,k), W1, W2, b, g_x);
        
    end
end
