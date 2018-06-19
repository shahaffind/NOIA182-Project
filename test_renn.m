dim = 5;
iter = 10;

X = randn(dim, 1);

b = randn(dim, 1);
W1 = randn(dim, dim);
W2 = randn(dim, dim);

d1 = randn(dim, 1);
d2 = randn(dim,dim);
d3 = randn(dim,dim);

vec_d = vertcat(d1(:), d2(:), d3(:));
eps = 1;

f = @(W1, W2, b, X) W2*X + arrayfun(@sigmoid, W1*X + b);

for i=1:iter
    eps = eps * 0.5;
    dist_f = norm(f(W1 + d2.*eps, W2 + d3.*eps, b + d1*eps, X) - f(W1,W2,b,X));
    grad = grad_ReNN_fortest(X, W1, W2, b, eps*vec_d);
    dist_g = norm(f(W1 + d2.*eps, W2 + d3.*eps, b + d1*eps, X) - f(W1,W2,b,X) - grad);
    disp([dist_f, dist_g]);
end  