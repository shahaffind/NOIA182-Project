function [ loss_val, Xis ] = forward( X, C, Theta, n_layers )
    
    [sample_size, n_samples] = size(X);
    Xis = zeros(sample_size, n_samples, n_layers);
    X_prev = X;
    for i=1:n_layers
        [W1, W2, b] = get_layer_weights(Theta, i, sample_size);
        Xis(:,:,i) = f(W1, W2, b, X_prev);
        X_prev = Xis(:,:,i);
    end
    
end


function [ W1, W2, b ] = get_layer_weights( Theta, layer, sample_size )
    theta_layer_size = (sample_size^2)*2 + sample_size;
    layer_weights = Theta((layer-1)*theta_layer_size + 1 : layer*theta_layer_size);
    W1 = layer_weights(1: sample_size^2);
    W2 = layer_weights(sample_size^2 + 1: 2*sample_size^2);
    b = layer_weights(2*sample_size^2 + 1:theta_layer_size);
end

function [ X_next ] = f( W1, W2, b, X_prev)
    X_next = X + W2 * sigmoid(X_prev, W1, b);
end
