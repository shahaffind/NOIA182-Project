load('.\data\GMMData.mat')

[n_classes, ~] = size(Ct);
[sample_size, n_samples] = size(Yt);

batch_size = 100;
max_epoch = 1000;

W = randn(sample_size, n_classes);
b = randn(n_classes,1);

theta = vertcat(b,W(:));

[theta_new, theta_all, iters] = loss_SGD(Yt, Ct, theta, batch_size, max_epoch);

all_proba = softmax(Yv, W, b);
loss_val = loss( Yv, Cv, W, b );
[c_p, R] = correct_percent(all_proba, Cv);
viewFeatures2D(Yv, R);
fprintf('iter:%d\n\tloss: %f\n\tcorrect: %f\n\n', i, loss_val, c_p);

for i = 1:iters
    theta_curr = theta_all(:,i);
    b_curr = theta_curr(1:n_classes);
    W_curr = reshape(theta_curr(n_classes+1:n_classes+sample_size*n_classes), [sample_size,n_classes]);
    
    all_proba = softmax(Yv, W_curr, b_curr);
    loss_val = loss( Yv, Cv, W_curr, b_curr );
    [c_p, R] = correct_percent(all_proba, Cv);
    viewFeatures2D(Yv, R);
    fprintf('iter:%d\n\tloss: %f\n\tcorrect: %f\n\n', i, loss_val, c_p);
end


% function [ correct_percent ] = calc_correct(Y, C, W)
%     [n_classes, n_samples] = size(C); 
%     
%     log_numer = Y' * W;
%     max_log_numer = max(log_numer, [], 2);
%     log_numer = bsxfun(@minus, log_numer, max_log_numer);
%     numerator = exp(log_numer);
%     denominator = sum(numerator, 2);
%     
%     all_proba = diag(denominator.^(-1)) * numerator;
%     
%     [~,idx_max] = max(all_proba', [], 1);
%     idx_max = idx_max + (0:n_samples-1) * n_classes;
%     R = zeros(n_classes, n_samples);
%     R(idx_max) = 1;
%     
%     correct = R .* C;
%     
%     correct_percent = sum(correct(:)) / n_samples;
% end