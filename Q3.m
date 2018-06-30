load('D:\NOIA-proj\data\PeaksData.mat')

Y = Yt;
C = Ct;

[n_classes, ~] = size(C);
[n_point_size, n_samples] = size(Y);

W = randn(n_point_size, n_classes);
 
[W_new, W_all, iters] = SGD(@loss_grad_theta, Y, C, W, 10, 10);

correct_percent = calc_correct(Y,C,W);
disp(correct_percent);

for i = 1:iters
    W_curr = W_all(:,:,i);
    
    correct_percent = calc_correct(Y,C,W_curr);
    disp(correct_percent);
end


function [ correct_percent ] = calc_correct(Y, C, W)
    [n_classes, n_samples] = size(C); 
    
    log_numer = Y' * W;
    max_log_numer = max(log_numer, [], 2);
    log_numer = bsxfun(@minus, log_numer, max_log_numer);
    numerator = exp(log_numer);
    denominator = sum(numerator, 2);
    
    all_proba = diag(denominator.^(-1)) * numerator;
    
    [~,idx_max] = max(all_proba', [], 1);
    idx_max = idx_max + (0:n_samples-1) * n_classes;
    R = zeros(n_classes, n_samples);
    R(idx_max) = 1;
    
    correct = R .* C;
    
    correct_percent = sum(correct(:)) / n_samples;
end