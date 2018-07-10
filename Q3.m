load('./data/PeaksData.mat')

[n_classes, ~] = size(Ct);
[sample_size, n_samples] = size(Yt);

batch_size = 100;
max_epoch = 100;
record_every = 1;
learning_rate = 0.005;

W = randn(sample_size, n_classes);
b = randn(n_classes,1);

theta = vertcat(b,W(:));

[theta_new, records, iters] = loss_SGD(Yt, Ct, theta, batch_size, learning_rate, max_epoch, Yv, Cv, record_every);

figure(99);
title('SGD on Loss')

xlabel('epoch');

yyaxis left;
plot(records(:,1));
ylabel('Loss');

yyaxis right;
plot(records(:,2));
ylabel('% correct');
