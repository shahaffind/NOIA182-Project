load('./data/PeaksData.mat')

% parameters
[n_labels, ~] = size(Ct);
[dim, ~] = size(Yt);
max_epoch = 200;
n_layers = 5;
batch_size = 50;

learning_rate = 0.01;
moment_gamma = 0.7;

record_every = 1;

theta_layer_size = dim + (dim^2) * 2;  % 2* NxN matrix + N vector
loss_layer_size = n_labels * (dim + 1);  % NxL matrix + L vector

% init all weights
Theta = randn(n_layers * theta_layer_size + loss_layer_size, 1);

% fit the data
[Theta_new, records] = ResNN_SGD(Yt, Ct, Theta, n_layers, batch_size, learning_rate, moment_gamma, max_epoch, Yv, Cv, record_every);

figure(99);
hold on
plot(records(:,1));

% figure(99);
% title('SGD on Network')
% 
% xlabel('epoch');
% 
% yyaxis left;
% plot(records(:,1));
% ylabel('Loss');
% 
% yyaxis right;
% plot(records(:,2));
% ylabel('% correct');
