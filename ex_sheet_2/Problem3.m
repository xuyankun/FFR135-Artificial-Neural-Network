clear, clc, clf, close all;

% --- IMPORTING DATA ---
addpath(genpath('Data'))
fileID = fopen('data_ex2_task3_2017.txt');
data = textscan(fileID,'%f %f %f');
fclose(fileID);

targets=data{1,1};

input_unsup=[data{1,2} data{1,3}];

kValues=10;
numRuns=1;
numUpdates1=1e5;
numUpdates2=3000;
p=size(input_unsup,1);
N=size(input_unsup,2);
meanCE=zeros(1,length(kValues));

for iKval=1:length(kValues)
    
    k=kValues(iKval);
    
    minCE=inf;
    CE=zeros(numRuns,1);
    
    for iRun=1:numRuns
        
        weights_unsup = -1*ones(k, N) +2*rand(k, N);
        
        for iUpdate=1:numUpdates1
            
            iPattern=randi([1 p]);
            weights_unsup = unsupervisedUpdate( input_unsup(iPattern,:), weights_unsup );
            
        end
        
        output_unsup = unsupervisedRun( input_unsup, weights_unsup );
        input_sup = output_unsup;
        
        weights_sup = -1*ones(1, k) +2*rand(1, k);
        threshold = -1*ones(1, 1) +2*rand(1, 1);
        
        for iUpdate=1:numUpdates2
            
            iPattern=randi([1 p]);
            [ weights_sup, threshold ] = supervisedUpdate( input_sup(iPattern,:), targets(iPattern), weights_sup, threshold );
            
        end
        
        output_sup = supervisedRun( input_sup, weights_sup, threshold );
        
        signO=sign(output_sup);
        signO(signO==0)=1;
        
        CE(iRun)=(1/(2*p))*sum(abs(targets-signO));
        
        if CE(iRun) < minCE
            
            minCE=CE(iRun);
            best_weights_unsup = weights_unsup;
            best_weights_sup = weights_sup;
            best_threshold = threshold;
            
        end
        
    end
    
    meanCE(iKval)=mean(CE);
    
end


disp(['Minimum classification error was: ' num2str(minCE)])

figure(1), hold on
for i=1:length(targets)
    if targets(i) == 1
        plot(input_unsup(i,1),input_unsup(i,2),'o','color','blue')
    else
        plot(input_unsup(i,1),input_unsup(i,2),'o','color','red')        
    end
end

numPointsX=50;
numPointsY=50;

[X,Y]=meshgrid(linspace(-15,25,numPointsX),linspace(-10,15,numPointsY));

for i=1:numPointsX
    for j=1:numPointsY
       
       input_unsup=[X(i,j) Y(i,j)];
       output_unsup = unsupervisedRun( input_unsup, best_weights_unsup );
       input_sup = output_unsup;
       output_sup = supervisedRun( input_sup, best_weights_sup, best_threshold );

       if output_sup >= 0
           plot(X(i,j),Y(i,j),'o','color','blue')
       else
           plot(X(i,j),Y(i,j),'o','color','red')           
       end
    
    end
end

for k=1:size(best_weights_unsup,1)
    quiver(0,0,best_weights_unsup(k,1),best_weights_unsup(k,2),'LineWidth',2.5,'MaxHeadSize',0.8,'color','black')
end

if length(kValues)>1
    figure(2)
    plot(kValues,meanCE)
end
    
    