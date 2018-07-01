load('.\data\GMMData.mat')

X = Yt;
C = Ct;

[n_labels, ~] = size(C);
[dim, ~] = size(X);
iter = 100;
n_layers = 4;
batch_size = 50;

theta_layer_size = dim + (dim^2) * 2;  % 2* NxN matrix + N vector
loss_layer_size = n_labels * (dim + 1);  % NxL matrix + L vector
Theta = randn(n_layers * theta_layer_size + loss_layer_size, 1);
 
[Theta_new, Theta_all, iters] = ResNN_SGD(X, C, Theta, n_layers, batch_size, iter);

[loss_val, ~] = forward_pass(Yv, Cv, Theta, n_layers);
disp(loss_val);

for i = 1:iters
    Theta_curr = Theta_all(:,i);
    [loss_val, ~] = forward_pass(Yv, Cv, Theta_curr, n_layers);
    disp(loss_val);
end

