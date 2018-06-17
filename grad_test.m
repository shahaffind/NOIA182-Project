function [ ] = grad_test( f, g, dim, iter )
    x = randn(dim, 1);
    d = randn(dim, 1);
    eps = 1;
    
    for i=1:iter
        eps = eps * 0.5;
        dist_f = abs(f(x + eps*d) - f(x));
        dist_g = abs(f(x + eps*d) - f(x) - eps*d'*g(x));
        disp([dist_f, dist_g]);
    end    
    
end

