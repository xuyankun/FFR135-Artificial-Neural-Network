clear;
clc;

p = 40; % p=5 for 2a, p=40 for 2b
N = 200;
t = 500;
beta = 2;
m20 = zeros(t,20);
iter = 20;

for k = 1:iter
    theta = 2 * randi([0,1],N,p) - 1 ;
    weight = zeros(N);
    
    for i = 1:p
       weightj = theta(:,i) * theta(:,i)'; 
       weight = weight + (1/N) .* weightj;
    end
    
    for i = 1:N
        weight(i,i) =0;
    end

    states = zeros(N,t);    
    m1 = zeros(t,1);

    state_0 = theta(:,1);
    states(:,1)= state_0;
    m1(1) = state_0' * state_0 / N;
    
for j=1:t-1
    b = weight * states(:,j);
    g = 1 ./ (1 + exp(-2 * beta .* b));

    for n = 1:N
       if rand(1,1)<g(n)
           states(n,j+1) = 1;
       else
           states(n,j+1) = -1;
       end
    end
    
    m1(j+1) = states(:,j+1)'*state_0 / N;
end

   m20(:,k) = m1;

end
m20_mean = mean(m20,2);
hold on
plot([1:t],m20(:,1:3));
plot([1:t],m20_mean,'b','linewidth',4)
xlabel('Time(Iterations)')
ylabel('m_{1}')
set(gca,'fontsize', 15)
ylim([-0.2 1.05]);











