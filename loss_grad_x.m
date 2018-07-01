function [ g ] = loss_grad_x( X, C, W, b )
    
    all_proba = softmax(X, W, b);
    
    g = W * (all_proba' - C);
end
