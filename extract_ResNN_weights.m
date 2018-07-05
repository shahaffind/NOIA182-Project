function [ W1, W2, b ] = extract_ResNN_weights( Theta, k, sample_size )
    % Theta - the entire net weights
    % k - the layer to extract
    % sample_size
    % 
    % returns W1, W2, b - the weights of the k-th layer

    
    theta_layer_size = sample_size + (sample_size^2) * 2;
    
    layer_weights = Theta((k - 1)*theta_layer_size + 1 : k*theta_layer_size);
    b = layer_weights(1 : sample_size);
    
    W1 = layer_weights(sample_size + 1 : sample_size + sample_size^2);
    W1 = reshape(W1, [sample_size,sample_size]);
    
    W2 = layer_weights(sample_size + sample_size^2 + 1 : theta_layer_size);
    W2 = reshape(W2, [sample_size,sample_size]);
end

