function [ g ] = grad_ReNN_fortest( x, W1, W2, b, v )
    [n,~] = size(W1);
    
    Jbv = mul_jac_b(x, W1, W2, b, v(1:n));
    Jw1v = mul_jac_W1(x, W1, W2, b, v(n+1:n+n^2));
    Jw2v = mul_jac_W2(x, W1, b, v(n+n^2+1:n+2*n^2));
    
    g = Jbv + Jw1v + Jw2v;
end


function [ res ] = mul_jac_b( x, W1, W2, b, v )
    inner = W1 * x + b;
    sig_inner = arrayfun(@sigmoid,inner);
    der_sig_inner = sig_inner .* (1-sig_inner);
    
    res = W2 * diag(der_sig_inner) * v;
end

function [ res ] = mul_jac_W1( x, W1, W2, b, v )
    [n, ~] = size(x);
    
    inner = W1 * x + b;
    sig_inner = arrayfun(@sigmoid,inner);
    der_sig_inner = sig_inner .* (1-sig_inner);
    left_side = W2 * diag(der_sig_inner);
    
    right_side = kron(x', eye(n));
    
    res = left_side * right_side * v;  % n times n*m
end

function [ res ] = mul_jac_W2( x, W1, b, v )
    [n, ~] = size(x);

    inner = W1 * x + b;
    sig_inner = arrayfun(@sigmoid,inner);
    
    res = kron(sig_inner', eye(n)) * v;  % n times n*m
end
