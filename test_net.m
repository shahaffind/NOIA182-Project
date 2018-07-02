load('.\data\SwissRollData.mat')

% parameters
[n_labels, ~] = size(Ct);
[dim, ~] = size(Yt);
iter = 50;
n_layers = 5;
batch_size = 500;

theta_layer_size = dim + (dim^2) * 2;  % 2* NxN matrix + N vector
loss_layer_size = n_labels * (dim + 1);  % NxL matrix + L vector

% init all weights
Theta = randn(n_layers * theta_layer_size + loss_layer_size, 1);

% fit the data
[Theta_new, Theta_all, iters] = ResNN_SGD(Yt, Ct, Theta, n_layers, batch_size, iter);

[all_proba, loss_val, ~] = forward_pass(Yv, Cv, Theta, n_layers);
[c_p, R] = correct_percent(all_proba, Cv);
viewFeatures2D(Yv, R);

fprintf("iter:0\n\tloss: %f\n\tcorrect: %f\n\n", loss_val, c_p);

for i = 1:iters
    Theta_curr = Theta_all(:,i);
    [all_proba, loss_val, ~] = forward_pass(Yv, Cv, Theta_curr, n_layers);
    [c_p, R] = correct_percent(all_proba, Cv);
    viewFeatures2D(Yv, R);
    
    fprintf("iter:%d\n\tloss: %f\n\tcorrect: %f\n\n", i, loss_val, c_p);
end

function[ c_p, R ] = correct_percent ( all_proba, C )
    
    [n_samples, n_classes] = size(all_proba);

    [~,idx_max] = max(all_proba', [], 1);
    idx_max = idx_max + (0:n_samples-1) * n_classes;
    R = zeros(n_classes, n_samples);
    R(idx_max) = 1;
    
    correct = R .* C;
    
    c_p = sum(correct(:)) / n_samples;
end
