function [ weights, threshold ] = supervisedUpdate( x, target, weights, threshold )
%supervisedUpdate: Summary of this function goes here
%   Detailed explanation goes here

beta=1/2;
eta=0.1;

g=@(b) tanh(beta*b);
gprim=@(b) beta*(1-tanh(beta*b).^2);

b=x*weights'-threshold;
O=g(b); % deriving output

delta=(target-O)*gprim(b); % calculating error

% updating weights
weights=weights+eta*delta*x;
threshold=threshold-eta*delta;

end

