function [ weights ] = unsupervisedUpdate( x, weights )
%unsupervisedUpdate: Summary of this function goes here
%   Detailed explanation goes here

k=size(weights,1);
eta=0.02;

diff=repmat(x,size(weights,1),1)-weights;
norms=sum(sqrt(diff(:,1).^2+diff(:,2).^2),2);
denominator = sum(exp(-0.5*norms.^2));

g=@(x,w,denominator) exp(-0.5*norm(x-w)^2)/denominator;

gMax=0;
i0=-1;
for i=1:k
   
   gCurrent=g(x,weights(i,:),denominator);
   
   if gCurrent > gMax
      
       gMax=gCurrent;
       i0=i;
       
   end
    
end

weights(i0,:)=weights(i0,:)+eta*(x-weights(i0,:));


end

