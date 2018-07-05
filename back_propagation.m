function [ g_theta ] = back_propagation( Xis, C, Theta, n_layers )
    % Xis - the values of the batch at each layer (+input layer at first)
    %       of size=(dim, batch_size, n_layers)
    % C - correct labels for each X_i in the batch
    % Theta - weights of the entire net
    % n_layers - number of layers in net
    % 
    % returns g_theta, the gradient of all the weights in the net
    
    
    [sample_size, ~] = size(Xis(:,:,1));
    [n_labels, ~] = size(C);
    
    % starting with the loss layer
    loss_weights_start_loc = n_layers*((sample_size^2)*2 + sample_size);
    loss_weights_end_loc = loss_weights_start_loc + n_labels +  sample_size * n_labels;
    [W_loss, b_loss] = extract_loss_weights(Theta, sample_size, n_labels, n_layers);
    
    g_theta = zeros(size(Theta));
    % stochastic gradient of loss by theta
    g_w_loss = loss_grad_theta(Xis(:,:,n_layers+1), C, W_loss, b_loss);
    g_theta(loss_weights_start_loc + 1 : loss_weights_end_loc) = g_w_loss;
    
    % exact gradient of loss by X to propagate to next layer
    % (for each col X_i, a respective col G_x_i)
    G_x = loss_grad_x(Xis(:,:,n_layers+1), C, W_loss, b_loss);
    
    theta_layer_size = sample_size + (sample_size^2) * 2;
    
    for k = n_layers:-1:1
        % get layer's weights
        [W1, W2, b] = extract_ResNN_weights(Theta, k, sample_size);
        
        % stochastic gradient of current layer by theta
        g_w_layer = ResNN_jac_theta_t_mul(Xis(:,:,k), W1, W2, b, G_x);
        g_theta((k-1)*theta_layer_size + 1 : k*theta_layer_size) = g_w_layer;
        
        % exact gradient of current layer by X to propagate to next layer
        % (for each col X_i, a respective col G_x_i)
        G_x = ResNN_jac_x_t_mul(Xis(:,:,k), W1, W2, b, G_x);
    end
end
