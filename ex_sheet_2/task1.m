clear;clc;clf;close all

% generate data
N_big = 1500;
N = 1000;
x1 = rand(N_big,1);
x2 = rand(N_big,1);
index = find(x1 > 0.5 & x2 <= 0.5);
x1(index) = [];
x2(index) = [];
data = [x1(1:N,:),x2(1:N,:)];

% initialization
T_order = 1e3;
sigma_0 = 100; % = 100 for 3a, = 5 for 3b 
lr_0 = 0.1;
tau_sigma = 300;

T_conv = 2e4;
sigma_conv = 0.9;
lr_conv = 0.01;

neurons = 100;
weights = 2*rand(2,neurons) - 1; % 2*100

% order phase
for t = 1:T_order
    sigma = sigma_0 * exp(-t/tau_sigma);
    lr = lr_0 * exp(-t/tau_sigma);
    
    mu = randi([1,N]);
    pattern = data(mu,:)'; % 2*1
    dist = zeros(1,neurons);
    for i = 1:neurons
        dist(i) = norm(pattern - weights(:,i));
    end
    win_index = find(dist == (min(dist)));
    
    delta_weights = zeros(size(weights));
    for j = 1:neurons
        delta_weights(:,j) = lr * exp(-0.5 * abs(j - win_index)^2/ sigma^2)...
            .* (pattern - weights(:,j));
    end
    weights = weights + delta_weights;
end

figure(1); hold on 
plot(weights(1,:),weights(2,:),'r-*','LineWidth',1)

% convergence phase
% weights = 2*rand(2,neurons) - 1;
for t = 1:T_conv
    sigma = sigma_conv;
    lr = lr_conv;
    
    mu = randi([1,1000]);
    pattern = data(mu,:)'; % 2*1
    dist = zeros(1,neurons);
    for i = 1:neurons
        dist(i) = norm(pattern - weights(:,i));
    end
    win_index = find(dist == (min(dist)));
    
    delta_weights = zeros(size(weights));
    for j = 1:neurons
        delta_weights(:,j) = lr * exp(-0.5 * abs(j - win_index)^2/ sigma^2)...
            .* (pattern - weights(:,j));
    end
    weights = weights + delta_weights;
end

figure(1);
plot(weights(1,:),weights(2,:),'b-*','LineWidth',1)
plot([0,0],[0,1],'k','LineWidth',1.5);plot([0.5,0.5],[0,0.5],'k','LineWidth',1.5);
plot([1,1],[0.5,1],'k','LineWidth',1.5);plot([0,0.5],[0,0],'k','LineWidth',1.5);
plot([0.5,1],[0.5,0.5],'k','LineWidth',1.5);plot([0,1],[1,1],'k','LineWidth',1.5);
title('Weight Space with $\sigma_{0}=100$','Interpreter','latex','FontSize',18)
xlabel('$w_1(\xi_1)$','Interpreter','latex','FontSize',14)
ylabel('$w_2(\xi_2)$','Interpreter','latex','FontSize',14)
legend('Weight vector after ordering phase','Weight vector after convergence phase','Location','southeast')
set(gca,'FontSize',15)












