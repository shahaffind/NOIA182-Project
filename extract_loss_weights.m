function [ W, b ] = extract_loss_weights( Theta, sample_size, n_labels, n_layers )

    loss_weights_loc = n_layers*((sample_size^2)*2 + sample_size);
    b = Theta(loss_weights_loc + 1 : loss_weights_loc + n_labels);
    W = Theta(loss_weights_loc + 1 + n_labels : loss_weights_loc + n_labels +  sample_size * n_labels);
    W = reshape(W, [sample_size, n_labels]);

end

