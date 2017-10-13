clear;clc;clf;close all

data = importdata('data_ex2_task2_2017.txt');

lr = 0.001;
time = 2e4;
weights = 2*rand(2,1) - 1;
weights_modulus = zeros(time,1);

for t = 1:time
    index = randi([1,size(data,1)]);
    pattern = data(index,:)'; % 2*1
    zeta = pattern' * weights; % 1*2 * 2*1
    delta_weights = lr * zeta .* (pattern - zeta .* weights);
    weights = weights + delta_weights;
    weights_modulus(t) = norm(weights);
end

figure(1); hold on
plot(1:time,weights_modulus,'Linewidth',2)

figure(2); hold on
plot(data(:,1),data(:,2),'rx','Linewidth',1)
quiver(0,0,weights(1),weights(2),'r','Linewidth',3,'MaxHeadSize',2)

% data with zero mean
data_zero_mean = [data(:,1) - mean(data(:,1)) , data(:,2) - mean(data(:,2))];
weights = 2*rand(2,1) - 1;
for t = 1:time
    index = randi([1,size(data_zero_mean,1)]);
    pattern = data_zero_mean(index,:)'; % 2*1
    zeta = pattern' * weights; % 1*2 * 2*1
    delta_weights = lr * zeta .* (pattern - zeta .* weights);
    weights = weights + delta_weights;
    weights_modulus(t) = norm(weights);
end

figure(1);
plot(1:time,weights_modulus,'Linewidth',2)
xlabel('Time(Iterations)')
ylabel('Modulus of weight vector')
legend('Curve of original data','Curve of data with zero mean')
set(gca,'FontSize',15)

figure(2); hold on
plot(data_zero_mean(:,1),data_zero_mean(:,2),'bx','Linewidth',1)
quiver(0,0,weights(1),weights(2),'k','Linewidth',3,'MaxHeadSize',2)
plot([-2 12],[0 0],'k','Linewidth',1.5);
plot([0 0],[-2 3],'k','Linewidth',1.5);
xlabel('$w_1(\xi_1)$','Interpreter','latex','FontSize',14)
ylabel('$w_2(\xi_2)$','Interpreter','latex','FontSize',14)
title('Data plot with ending weight vector','FontSize',18)
legend('Original data','Ending weight vector of oringinal data',...
    'Data with zero mean','Ending weight vector of oringinal data',...
    'Location','southeast')
set(gca,'FontSize',15)
xlim([-2,12])
ylim([-1.5,3])
