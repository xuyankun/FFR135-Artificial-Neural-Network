clc;
clear;

train_data = importdata('train_data_2017.txt');
valid_data = importdata('valid_data_2017.txt');

train_input = train_data(:,1:2);
train_target = train_data(:,3);
valid_input = valid_data(:,1:2);
valid_target = valid_data(:,3);

% generalize
for i = 1:2
   train_input(:,i) = (train_input(:,i)-...
       mean(train_input(:,i)))/std(train_input(:,i));
   valid_input(:,i) = (valid_input(:,i)-...
       mean(valid_input(:,i)))/std(valid_input(:,i));
end

% parameters
iter = 1e6;
lr = 0.02; % learning_rate
beta = 0.5;
runs = 10;

%% 3a
% initialize
energy_t_a = zeros(iter,1);
energy_v_a = zeros(iter,1);
C_t_a = zeros(runs,1);
C_v_a = zeros(runs,1);
weight_10 = zeros(2,runs);
bias_10 = zeros(1,runs);

for r = 1:runs
    
    weight = 0.4*rand(2,1)-0.2;
    bias = 2*rand(1,1)-1;
    
    for i = 1:iter
    % feed pattern
    index = randi([1 size(train_input,1)]);
    feed_pattern = train_input(index,:)';
    pattern_traget = train_target(index);
    
    b = weight' * feed_pattern - bias; 
    output = tanh(beta*b);

    % update
    weight = weight + lr*beta*(1-output^2)*(pattern_traget - output) .* feed_pattern;
    bias = bias - lr*beta*(1-output^2)*(pattern_traget - output); 

    output_t = tanh(beta.*(train_input*weight - bias));
    energy_t_a(i) = 0.5 * sum((train_target - output_t).^2);
    
    output_v = tanh(beta.*(valid_input*weight - bias));
    energy_v_a(i) = 0.5 * sum((valid_target - output_v).^2);
    end
    weight_10(:,r) = weight; 
    bias_10(:,r) = bias;
    % classification error of training and validation set
    C_t_a(r) = 0.5 * sum(abs(train_target-sign(output_t))) / size(train_data,1);
    C_v_a(r) = 0.5 * sum(abs(valid_target-sign(output_v))) / size(valid_data,1);
    
end

% analyze classification error
mean_ct_a = mean(C_t_a); % mean of C in training set
var_ct_a = var(C_t_a); % variance of C in training set
mini_ct_a = min(C_t_a); % minimum C in training set
mean_cv_a = mean(C_v_a); % mean of C in validation set
var_cv_a = var(C_v_a); % variance of C in validation set
mini_cv_a = min(C_v_a); % minimum C in validation set

%% plot
plot([1:iter],[energy_t_a energy_v_a]);
legend('Training set','Validation set')
xlabel('Iterations')
ylabel('Energy')
set(gca,'fontsize', 15)
% axis([1 runs 0 1])

%% 3b
energy_t_b = zeros(iter,1);
energy_v_b = zeros(iter,1);
C_t_b = zeros(runs,1);
C_v_b = zeros(runs,1);
weight_input_hidden_10 = zeros(2,4,runs);
bias_input_hidden_10 = zeros(4,runs);
for r = 1:runs
    
    % initialize weight and bias
    % we have 4 neurons in hidden layer
    weight_input_hidden = 0.4*rand(2,4)-0.2; 
    weight_hidden_output = 0.4*rand(4,1)-0.2; 

    bias_input_hidden = 2*rand(4,1)-1;
    bias_hidden_output = 2*rand(1,1)-1;
    
    for i = 1:iter
        % feed pattern
        index = randi([1 size(train_input,1)]);
        feed_pattern = train_input(index,:)';% 2*1
        pattern_traget = train_target(index);
        
        bj = weight_input_hidden' * feed_pattern - bias_input_hidden; 
        V = tanh(beta*bj); % output of hidden layer 4*1
        bi = weight_hidden_output' * V - bias_hidden_output;
        output = tanh(beta*bi);
        
        % update
        weight_input_hidden = weight_input_hidden + lr*beta*...
            beta*(1-output^2)*(pattern_traget - output).* ...
            feed_pattern*(weight_hidden_output.*(1-V.^2))';
        bias_input_hidden = bias_input_hidden - ...
            lr*beta*(1-V.^2)*beta*(1-output^2)*...
            (pattern_traget - output).*weight_hidden_output;
        weight_hidden_output = weight_hidden_output + lr*beta*...
            (1-output^2)*(pattern_traget - output) .* V;
        bias_hidden_output = bias_hidden_output - ...
            lr*beta*(1-output^2)*(pattern_traget - output); 
        
        V_t = tanh(beta.*(weight_input_hidden'*train_input' - bias_input_hidden));
        output_t = tanh(beta.*(weight_hidden_output'*V_t - bias_hidden_output));
        energy_t_b(i) = 0.5 * sum((train_target - output_t').^2);

        V_v = tanh(beta.*(weight_input_hidden'*valid_input' - bias_input_hidden));
        output_v = tanh(beta.*(weight_hidden_output'*V_v - bias_hidden_output));
        energy_v_b(i) = 0.5 * sum((valid_target - output_v').^2);
    
    end
    
    weight_input_hidden_10(:,:,r) = weight_input_hidden;
    bias_input_hidden_10(:,r) = bias_input_hidden;
    % classification error of training and validation set
    C_t_b(r) = 0.5 * sum(abs(train_target-sign(output_t'))) / size(train_data,1);
    C_v_b(r) = 0.5 * sum(abs(valid_target-sign(output_v'))) / size(valid_data,1);
    
end

%analyze classification error
mean_ct_b = mean(C_t_b); % mean of C in training set
var_ct_b = var(C_t_b); % variance of C in training set
mini_ct_b = min(C_t_b); % minimum C in training set
mean_cv_b = mean(C_v_b); % mean of C in validation set
var_cv_b = var(C_v_b); % variance of C in validation set
mini_cv_b = min(C_v_b); % minimum C in validation set

%% plot
plot([1:iter],[energy_t_b energy_v_b]);
legend('Training set','Validation set')
xlabel('Iterations')
ylabel('Energy')
set(gca,'fontsize', 15)
% axis([1 runs 0 1])

%% classification plot 

class1 = valid_input(valid_target == 1,:);
class2 = valid_input(valid_target == -1,:);

x = linspace(-2,2,100);
% boundary in 3a 
weight = weight_10(:,find(C_v_a==mini_cv_a));
bias = bias_10(:,find(C_v_a==mini_cv_a));
line = bias/weight(2) - x * weight(1)/weight(2) ; % weight' * input = bias
subplot(1,2,1)
hold on 
plot(class1(:,1),class1(:,2),'ro',class2(:,1),class2(:,2),...
    'bx','MarkerSize',10,'linewidth',2);
plot(x,line,'linewidth',2.5);
legend('Class +1',' Class -1', 'Boundary')
set(gca,'fontsize', 15)
xlabel('\xi_1')
ylabel('\xi_2')
ylim([-2 2])

% boundary in 3a
% we have 4 lines due to 4 neurons in the hidden layer
weight_input_hidden = weight_input_hidden_10(:,:, find(mini_cv_b == C_v_b));
bias_input_hidden = bias_input_hidden_10(:,find(mini_cv_b == C_v_b));

lines = zeros(4,length(x));

for l = 1:4
    lines(l,:) = bias_input_hidden(l)/weight_input_hidden(2,l) - ...
    x .* weight_input_hidden(1,l)/weight_input_hidden(2,l);
    
end

subplot(1,2,2)
hold on 
plot(class1(:,1),class1(:,2),'ro',class2(:,1),class2(:,2),...
    'bx','MarkerSize',10,'linewidth',2);
plot(x,lines,'linewidth',2.5);
legend('Class +1',' Class -1', 'Boundary')
set(gca,'fontsize', 15)
xlabel('\xi_1')
ylabel('\xi_2')
ylim([-2 2])







