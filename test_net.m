C = [1;0;0;0;0];
[n_labels, n_samples] = size(C);
dim = 3;
X = randn(dim, n_samples);

n_layers = 1;

theta_layer_size = dim + (dim^2) * 2;  % 2* NxN matrix + N vector
loss_layer_size = n_labels * (dim + 1);  % NxL matrix + L vector

% init all weights
Theta = randn(n_layers * theta_layer_size + loss_layer_size, 1);

iter = 20;
d_Theta = randn(n_layers * theta_layer_size + loss_layer_size, 1);

eps = 1;

dist_f = inf;
dist_g = inf;

% forward
[~, base_loss_val, Xis] = forward_pass(X, C, Theta, n_layers);

% backward
grad = back_propagation(Xis, C, Theta, n_layers);

for i=1:iter
    eps = eps * 0.5;
    
    [~, loss_val, ~] = forward_pass(X, C, Theta + eps*d_Theta, n_layers);
    
    curr_dist_f = abs(loss_val - base_loss_val);
    curr_dist_g = abs(loss_val - base_loss_val - eps*d_Theta'*grad);
    
    disp([dist_f / curr_dist_f, dist_g / curr_dist_g]);
    dist_f = curr_dist_f;
    dist_g = curr_dist_g;
end

