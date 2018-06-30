function [ g_theta ] = backpropagation( X, C, Theta, Xis, n_layers )
    
    [sample_size, n_samples] = size(X);
    [n_labels, ~] = size(C);
    
    theta_layer_size = sample_size + (sample_size^2) * 2; % 2* NxN matrix + N vector
    
    loss_weights_loc = n_layers*theta_layer_size + 1;
    W_loss = Theta(loss_weights_loc : loss_weights_loc + n_labels);
    
    g_theta = zeros(size(Theta));
    g_theta(loss_weights_loc : loss_weights_loc + n_labels) = loss_grad_theta(Xis(:,:,n_layers), C, W_loss);
    
    g_x = loss_grad_x(Xis(:,:,n_layers), C, W_loss)
    
    for k = n_layers:-1:1
        layer_weights = Theta((layer-1)*theta_layer_size + 1 : layer*theta_layer_size);
        b = layer_weights(1 : sample_size);
        W1 = layer_weights(sample_size + 1 : sample_size + sample_size^2);
        W2 = layer_weights(sample_size + sample_size^2 + 1 : theta_layer_size);
        
        X_prev = Xis(:,:,k - 1);
        
        g_w_layer = ResNN_jac_theta_t_mul(X_prev, W1, W2, b, g_x);
        
        g_theta(loss_weights_loc + (k-1)*theta_layer_size + 1 : loss_weights_loc + k*theta_layer_size) = g_w_layer;
        
        g_x = ResNN_jac_x_t_mul(X_prev, W1, W2, b, g_x);
        
    end
    

end
