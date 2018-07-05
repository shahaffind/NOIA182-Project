function [ all_proba, loss_val, Xis ] = forward_pass( X, C, Theta, n_layers )
    % X - batch with size=(dim, batch_size)
    % C - correct labels for each X_i in the batch
    % Theta - weights of the entire net
    % n_layers - number of layers in net
    % 
    % returns: all proba - for each X_i, the probability for it to be of
    %                      each of the classes
    %          loss_val - the loss of the net
    %          Xis - the values of the batch at each layer
    %                (+input layer at first)
    
    
    [sample_size, n_samples] = size(X);
    [n_labels, ~] = size(C);
    
    Xis = zeros(sample_size, n_samples, n_layers+1);
    Xis(:,:,1) = X;  % first Xis is the input layer
    for k=1:n_layers
        [W1, W2, b] = extract_ResNN_weights(Theta, k, sample_size);
        
        % use last layer's results to pass current layer, save results
        Xis(:,:,k+1) = ResNN(Xis(:,:,k), W1, W2, b);
    end
    
    loss_weights_loc = n_layers*((sample_size^2)*2 + sample_size);
    b = Theta(loss_weights_loc + 1 : loss_weights_loc + n_labels);
    W = Theta(loss_weights_loc + 1 + n_labels : loss_weights_loc + n_labels +  sample_size * n_labels);
    W = reshape(W, [sample_size, n_labels]);
    
    % calculate the loss and probabilities of the net for 
    loss_val = loss(Xis(:,:,n_layers+1), C, W, b);
    all_proba = softmax(Xis(:,:,n_layers+1), W, b);
end
