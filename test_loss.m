sample_size = 5;
iter = 10;

C = [1,0,0,0,0,1,0,0,0,1;
     0,1,0,1,0,0,1,0,1,0;
     0,0,1,0,1,0,0,1,0,0;];

%C = [1;0;0];

[n_labels, n_samples] = size(C);

X = randn(sample_size, n_samples);
W = randn(sample_size, n_labels);
b = randn(n_labels, 1);

d_w = randn(sample_size,n_labels);
d_b = randn(n_labels, 1);
d = vertcat(d_b(:), d_w(:));

d_x = randn(sample_size, n_samples);

eps = 1;

dist_f = inf;
dist_g = inf;

for i=1:iter
    eps = eps * 0.5;
    %%%% grad w.r.t weights
    %curr_dist_f = abs(loss(X,C,W + d_w.*eps, b + d_b.*eps) - loss(X,C,W,b));
    %grad = loss_grad_theta(X,C,W,b);
    %curr_dist_g = abs(loss(X,C,W + d_w.*eps, b + d_b.*eps) - loss(X,C,W,b) - eps*d'*grad);
    
    %%%% grad w.r.t x
    curr_dist_f = abs(loss(X + d_x.*eps,C,W,b) - loss(X,C,W,b));
    grad = loss_grad_x(X,C,W,b);
    curr_dist_g = abs(loss(X + d_x.*eps,C,W,b) - loss(X,C,W,b) - eps*d_x(:)'*grad(:));
    
    disp([dist_f / curr_dist_f, dist_g / curr_dist_g]);
    dist_f = curr_dist_f;
    dist_g = curr_dist_g;
end  