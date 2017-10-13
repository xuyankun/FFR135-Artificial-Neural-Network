function [ O ] = supervisedRun( x_set, weights, threshold )
%supervisedRun: Summary of this function goes here
%   Detailed explanation goes here

p=size(x_set,1);
beta=1/2;

g=@(b) tanh(beta*b);

b=zeros(p,1);
for i=1:p
    b(i)=x_set(i,:)*weights'-threshold;
end

O=g(b); % deriving output


end

