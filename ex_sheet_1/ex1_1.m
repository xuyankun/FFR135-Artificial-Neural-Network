clear ;
clc;

p = [1 linspace(20,400,20)]; 
N = 200; % number of nuerons 
bits = 100000;
P_error = zeros(length(p),1);
error_bits = zeros(length(p),1);

for i = 1 : length(p)
    
    iter = round( bits / (p(i) * N));
    error_vector = zeros( p(i)*N , iter);
    
    for k = 1:iter
        patterns = 2 * randi([0,1],N,p(i)) - 1 ;
        weight = zeros(N);
    
        for j = 1:p(i)
            weightj = patterns(:,j) * patterns(:,j)'; % weight of each pattern
            weight = weight + (1/N) .* weightj;
        end
        
        for m = 1:p(i)
            S_j = patterns(:,m);
            S_i = sign(weight * S_j);
            S_i = S_i + (S_i==0) .* (2*randi([0,1],N,1)-1);
            error_vector(((m-1)*N+1):m*N,k) = S_i ~= S_j;
        end
    end
    error_vector = error_vector(:);
    index = randperm(length(error_vector));
    index = index(1:iter*p(i)*N);
    error_bits(i) = sum(error_vector(index));
    P_error(i) = error_bits(i) / length(index);
    
end

p2 = [1:400];
P_error1 = 0.5 * (1 - erf( (1 + p2./N) ./ sqrt( 2 * p2 ./ N)));

plot( p ./ N ,P_error,'r--*',p2 ./ N,P_error1,'Markersize',8,'linewidth',1.5);
xlabel('\alpha = p/N');
ylabel('P(Error)');
legend('Expermental curve','Theoratical curve','Location','northwest')
set(gca,'fontsize', 15)
