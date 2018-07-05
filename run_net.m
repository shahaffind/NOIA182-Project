load('./data/PeaksData.mat')

% parameters
[n_labels, ~] = size(Ct);
[dim, ~] = size(Yt);
max_epoch = 10000;
n_layers = 15;
batch_size = 50;

record_every = 10;

theta_layer_size = dim + (dim^2) * 2;  % 2* NxN matrix + N vector
loss_layer_size = n_labels * (dim + 1);  % NxL matrix + L vector

% init all weights
Theta = randn(n_layers * theta_layer_size + loss_layer_size, 1);

% fit the data
[Theta_new, Theta_all, iters] = ResNN_SGD(Yt, Ct, Theta, n_layers, batch_size, max_epoch, Yv, Cv, record_every);
