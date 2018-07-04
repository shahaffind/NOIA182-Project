function [ all_proba, loss_val, Xis ] = forward_pass( X, C, Theta, n_layers )
    
    [sample_size, n_samples] = size(X);
    [n_labels, ~] = size(C);
    
    Xis = zeros(sample_size, n_samples, n_layers+1);
    Xis(:,:,1) = X;
    for k=1:n_layers
        [W1, W2, b] = get_layer_weights(Theta, k, sample_size);
        Xis(:,:,k+1) = ResNN(Xis(:,:,k), W1, W2, b);
    end
    
    loss_weights_loc = n_layers*((sample_size^2)*2 + sample_size);
    b = Theta(loss_weights_loc + 1 : loss_weights_loc + n_labels);
    W = Theta(loss_weights_loc + 1 + n_labels : loss_weights_loc + n_labels +  sample_size * n_labels);
    W = reshape(W, [sample_size, n_labels]);
    
    loss_val = loss(Xis(:,:,n_layers+1), C, W, b);
    all_proba = softmax(Xis(:,:,n_layers+1), W, b);
end

function [ W1, W2, b ] = get_layer_weights( Theta, layer, sample_size )
    theta_layer_size = (sample_size^2)*2 + sample_size;
    layer_weights = Theta((layer-1)*theta_layer_size + 1 : layer*theta_layer_size);
    b = layer_weights(1 : sample_size);
    W1 = layer_weights(sample_size + 1 : sample_size + sample_size^2);
    W2 = layer_weights(sample_size + sample_size^2 + 1 : theta_layer_size);
    W1 = reshape(W1, [sample_size,sample_size]);
    W2 = reshape(W2, [sample_size,sample_size]);
end
