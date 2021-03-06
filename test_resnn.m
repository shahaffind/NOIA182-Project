dim = 5;
iter = 20;

X = randn(dim, 1);

b = randn(dim, 1);
W1 = randn(dim, dim);
W2 = randn(dim, dim);

dx = randn(dim, 1);
d1 = randn(dim, 1);
d2 = randn(dim,dim);
d3 = randn(dim,dim);
vec_d = vertcat(d1(:), d2(:), d3(:));
eps = 1;

dist_f = inf;
dist_g = inf;

disp('jacobian test (w.r.t theta)');

for i=1:iter
    eps = eps * 0.5;
    
    %%%% grad w.r.t weights
    curr_dist_f = norm(ResNN(X, W1 + d2.*eps, W2 + d3.*eps, b + d1*eps) - ResNN(X,W1,W2,b));
    grad = ResNN_jac_theta_mul(X, W1, W2, b, eps*vec_d);
    curr_dist_g = norm(ResNN(X, W1 + d2.*eps, W2 + d3.*eps, b + d1*eps) - ResNN(X,W1,W2,b) - grad);
    
    disp([dist_f / curr_dist_f, dist_g / curr_dist_g]);
    dist_f = curr_dist_f;
    dist_g = curr_dist_g;
end

 disp('jacobian transpose test (w.r.t theta)');
 
for i = 1:iter
    %%%% jac w.r.t weights
    u = randn(size(vec_d));
    v = randn(dim, 1);
    jac = ResNN_jac_theta_mul(X, W1, W2, b, u);
    jac_t = ResNN_jac_theta_t_mul(X, W1, W2, b, v);
    disp(abs(v' * jac - u' * jac_t));
end

disp('jacobian test (w.r.t x)');

eps = 1;
dist_f = inf;
dist_g = inf;

for i=1:iter
    eps = eps * 0.5;
    
    %%%% grad w.r.t. X
    curr_dist_f = norm(ResNN(X + dx*eps,W1, W2, b) - ResNN(X,W1,W2,b));
    grad = ResNN_jac_x_mul(X, W1, W2, b, eps*dx);
    curr_dist_g = norm(ResNN(X + dx*eps,W1, W2, b) - ResNN(X,W1,W2,b) - grad);
    
    disp([dist_f / curr_dist_f, dist_g / curr_dist_g]);
    dist_f = curr_dist_f;
    dist_g = curr_dist_g;
end

 disp('jacobian transpose test (w.r.t x)');
 
for i = 1:iter
    
    %%%% jac w.r.t X
    u = randn(dim,1);
    v = randn(dim,1);
    jac = ResNN_jac_x_mul(X, W1, W2, b, u);
    jac_t = ResNN_jac_x_t_mul(X, W1, W2, b, v);
    disp(abs(v' * jac - u' * jac_t));
end

