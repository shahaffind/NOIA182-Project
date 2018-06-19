dim = 5;
iter = 10;

X = randn(dim, 10);
W = randn(dim, dim);
C = [1,0,0,0,0,1,0,0,0,0;
     0,1,0,0,0,0,1,0,1,0;
     0,0,1,0,0,0,0,1,0,0;
     0,0,0,1,0,0,0,0,0,1;
     0,0,0,0,1,0,0,0,0,0];

d = randn(dim,dim);
eps = 1;

for i=1:iter
    eps = eps * 0.5;
    dist_f = abs(loss(X,C,W + d.*eps) - loss(X,C,W));
    grad = grad_loss(X,C,W);
    dist_g = abs(loss(X,C,W + d.*eps) - loss(X,C,W) - eps*d(:)'*grad(:));
    disp([dist_f, dist_g]);
end  