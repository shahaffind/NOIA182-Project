load('./data/PeaksData.mat')

[n_classes, ~] = size(Ct);
[sample_size, n_samples] = size(Yt);

batch_size = 50;
max_epoch = 10000;
record_every = 10;

W = randn(sample_size, n_classes);
b = randn(n_classes,1);

theta = vertcat(b,W(:));

[theta_new, theta_all, iters] = loss_SGD(Yt, Ct, theta, batch_size, max_epoch, Yv, Cv, record_every);
