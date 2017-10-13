function [ O ] = unsupervisedRun( x_set, weights )
%unsupervisedRun: Summary of this function goes here
%   Detailed explanation goes here

p=size(x_set,1);
k=size(weights,1);

g=@(x,w,denominator) exp(-0.5*norm(x-w)^2)/denominator;

O=zeros(p,k);

for i=1:p
    
    diff=repmat(x_set(i,:),size(weights,1),1)-weights;
    norms=sum(sqrt(diff(:,1).^2+diff(:,2).^2),2);
    denominator = sum(exp(-0.5*norms.^2));
    
    for j=1:k
        
        O(i,j)=g(x_set(i,:),weights(j,:),denominator);
        
    end 
end

end

