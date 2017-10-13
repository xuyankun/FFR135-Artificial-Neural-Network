clear;clc;clf;close all

data = importdata('data_ex2_task3_2017.txt');
runs = 20;

k=10; % k=4 for 3a , k=10 for 3b

class_error_runs = zeros(runs,1);
weights_unsup_runs = zeros(2*runs,k);
weights_sup_runs = zeros(k,runs);
bias_runs = zeros(runs,1);
for r = 1:runs
    % unsupervised learning parameters
    lr_uns = 0.02;
    time = 1e5;
    weights_unsupervised = 2*rand(2,k) - 1; % 2*k
    
    % supervised learning parameters
    lr_s = 0.1;
    beta = 0.5;
    steps = 3e3;
    weights_supervised = 2*rand(k,1) - 1; % k*1
    bias = 2*rand(1,1)-1;
    
    % unsupervised learning
    for t = 1:time
        index_pattern = randi([1 size(data,1)]);
        feed_pattern = data(index_pattern,2:3)'; % 2*1
        activation = zeros(k,1);
        
        for j = 1:k
            activation(j) = exp(- norm(feed_pattern - weights_unsupervised(:,j))^2 /2 );
        end
        
        activation = activation ./ sum(activation);
        index_win_unit = find(activation == max(activation));
        delta_win_weights = lr_uns * (feed_pattern - weights_unsupervised(:,index_win_unit));
        weights_unsupervised(:,index_win_unit) = weights_unsupervised(:,index_win_unit) + delta_win_weights;
    end
    weights_unsup_runs(2*(r-1)+1:r*2,:) = weights_unsupervised;
    % supervised simple perceptron learning
    
    
    for s = 1:steps
        % feed pattern
        index_SGD = randi([1 size(data,1)]);
        feed_pattern = data(index_SGD,2:3)';
        feed_target = data(index_SGD,1);
        perceptron_neurons = zeros(4,1);
        
        for j = 1:k
            perceptron_neurons(j) = exp(- norm(feed_pattern - weights_unsupervised(:,j))^2 /2 );
        end
        
        perceptron_neurons = perceptron_neurons ./ sum(perceptron_neurons);
        b = weights_supervised' * perceptron_neurons - bias;
        output = tanh(beta * b);
        
        % BP update
        weights_supervised = weights_supervised + lr_s*beta*(1-output^2)*...
            (feed_target - output) .* perceptron_neurons;
        bias = bias - lr_s*beta*(1-output^2)*(feed_target - output);    
    end
    weights_sup_runs(:,r) = weights_supervised;
    bias_runs(r) = bias;
       
    num_error = 0;
    for i = 1:size(data,1)
        pattern = data(i,2:3)'; %2*1
        target = data(i,1);
        output_unsup = zeros(k,1);
        for j = 1:k
            output_unsup(j) = exp(- norm(pattern - weights_unsupervised(:,j))^2 / 2 );
        end
        output_unsup = output_unsup ./ sum(output_unsup); % 4*1
        output = tanh(beta*(weights_supervised' * output_unsup - bias));
        if sign(output) ~= target
            num_error = num_error + 1;
        end
    end
    class_error_runs(r) = num_error / size(data,1) ;
end


%% decision boundary
best_index = find(class_error_runs == min(class_error_runs));
best_index = best_index(1);
best_unsup_weights = weights_unsup_runs((best_index-1)*2+1:2*best_index,:);
best_sup_weights = weights_sup_runs(:,best_index);
best_bias = bias_runs(best_index);

num_points = 100;
[X,Y] = meshgrid(linspace(-15,25,num_points),linspace(-10,15,num_points));
decision1 = [];
decision2 = [];
for x = 1:num_points
    for y = 1:num_points
        input_pattern = [X(x,y);Y(x,y)]; % 2*1
        output_unsup = zeros(k,1);
        for j = 1:k
            output_unsup(j) = exp(- norm(input_pattern - best_unsup_weights(:,j))^2 / 2 );
        end
        output_unsup = output_unsup ./ sum(output_unsup); % 4*1
        output = tanh(beta*(best_sup_weights' * output_unsup - bias));
        if output >= 0
            decision1 = cat(1,decision1,[X(x,y) Y(x,y)]);
        else
            decision2 = cat(1,decision2,[X(x,y) Y(x,y)]);
        end
    end    
end

%% plot
class1 = data(data(:,1) == 1, 2:3);
class2 = data(data(:,1) == -1, 2:3);
figure(1); hold on
plot(decision1(:,1),decision1(:,2),'g*','Linewidth',10)
plot(decision2(:,1),decision2(:,2),'y*','Linewidth',10)
plot(class1(:,1), class1(:,2), 'ro','Linewidth',1.5);
plot(class2(:,1), class2(:,2), 'bx','Linewidth',1.5);
title('Data plot with decision boundary when k=10','FontSize',18)
xlabel('$\xi_1$','Interpreter','latex','FontSize',14)
ylabel('$\xi_2$','Interpreter','latex','FontSize',14)
legend('Decision boundary of class = 1','Decision boundary of class = -1',...
    'Data of class = 1','Data of class = -1','Location','northeast')
set(gca,'FontSize',15)












