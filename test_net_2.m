C = [1;0;0;0;0];
[n_labels, n_samples] = size(C);
dim = 3;
X = randn(dim, n_samples);

iter = 20;

W_loss = randn(dim, n_labels);
b_loss = randn(n_labels,1);

t_loss = vertcat(b_loss(:), W_loss(:));

W1_2 = randn(dim,dim);
W2_2 = randn(dim,dim);
b_2 = randn(dim, 1);

t_r2 = vertcat(b_2(:), W1_2(:), W2_2(:));

W1 = randn(dim,dim);
W2 = randn(dim,dim);
b = randn(dim, 1);

t_r1 = vertcat(b(:), W1(:), W2(:));

Theta = vertcat(t_r1, t_r2, t_loss);

d_w_l = randn(dim, n_labels);
d_b_l = randn(n_labels,1);

d_loss = vertcat(d_b_l(:), d_w_l(:));

d_w1_2 = randn(dim,dim);
d_w2_2 = randn(dim,dim);
d_b_2 = randn(dim, 1);

d_r2 = vertcat(d_b_2(:), d_w1_2(:), d_w2_2(:));

d_w1 = randn(dim,dim);
d_w2 = randn(dim,dim);
d_b = randn(dim, 1);

d_r1 = vertcat(d_b(:), d_w1(:), d_w2(:));

d = vertcat(d_r1, d_r2, d_loss);

eps = 1;

dist_f = inf;
dist_g = inf;

% forward
X_mid = ResNN(X, W1, W2, b); % f1(X, theta1)
X_mid2 = ResNN(X_mid, W1_2, W2_2, b_2); % f2(f1(X, theta1), theta2)
base_loss_val = loss(X_mid2, C, W_loss, b_loss); % f3(f2(f1(X, theta1), theta2), theta2)

% backward
grad_l_theta = loss_grad_theta(X_mid2, C, W_loss, b_loss); % g_theta2_f3
grad_l_x = loss_grad_x(X_mid2, C, W_loss, b_loss); % g_x_f3
grad_r2_theta = ResNN_jac_theta_t_mul(X_mid, W1_2, W2_2, b_2, grad_l_x); % jac_theta2_trans*g_x_f3
grad_r2_x = ResNN_jac_x_t_mul(X_mid, W1_2, W2_2, b_2, grad_l_x); % g_x_f2 = jac_x_trans*g_x_f3
grad_r1_theta = ResNN_jac_theta_t_mul(X, W1, W2, b, grad_r2_x); % jac_theta1_trans*g_x_f2

grad = vertcat(grad_r1_theta, grad_r2_theta, grad_l_theta);

for i=1:iter
    eps = eps * 0.5;
    
    X_mid_new = ResNN(X, W1+d_w1*eps, W2+d_w2*eps, b+d_b*eps); % f1(X, theta1)
    X_mid2_new = ResNN(X_mid_new, W1_2+d_w1_2*eps, W2_2+d_w2_2*eps, b_2+d_b_2*eps); % f2(f1(X, theta1), theta2)
    loss_val = loss(X_mid2_new, C, W_loss+d_w_l*eps, b_loss+d_b_l*eps); % f3(f2(f1(X, theta1), theta2), theta2)
    
    curr_dist_f = abs(loss_val - base_loss_val);
    curr_dist_g = abs(loss_val - base_loss_val - eps*d'*grad);
    
    disp([dist_f / curr_dist_f, dist_g / curr_dist_g]);
    dist_f = curr_dist_f;
    dist_g = curr_dist_g;
end
