function [Accuracy,Training_time] = oabls_train(X_Train,Y_Train,X_Test,Y_Test,s,C,N1,N2,N3)

% Learning Process of the proposed broad learning system

%% feature nodes Creation
tic

H1 = [X_Train .1 * ones(size(X_Train,1),1)];

y=zeros(size(X_Train,1),N2*N1);

for i=1:N2
    we=2*rand(size(X_Train,2)+1,N1)-1;
    We{i}=we;
    A1 = H1 * we;A1 = mapminmax(A1);
    clear we;
beta1  =  sparse_bls(A1,H1,1e-3,50)';

beta11{i}=beta1;

% clear A1;

T1 = H1 * beta1;


[T1,ps1]  =  mapminmax(T1',0,1);T1 = T1';

ps(i)=ps1;

% clear H1;y=[y T1];

y(:,N1*(i-1)+1:N1*i)=T1;
end

clear H1;
clear T1;
%% Enhancement NodeS Creation

H2 = [y .1 * ones(size(y,1),1)];

if N1*N2>=N3
     wh=orth(2*rand(N2*N1+1,N3)-1);
else
    wh=orth(2*rand(N2*N1+1,N3)'-1)'; 
end

T2 = H2 *wh;

l2 = max(max(T2));

l2 = s/l2;


T2 = tansig(T2 * l2);

T3=[y T2];

clear H2;clear T2;

beta = (T3'  *  T3+eye(size(T3',1)) * (C)) \ ( T3'  *  Y_Train);

Training_time = toc;

disp('Training has been finished!');



%%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%

xx = T3 * beta;

clear T3;

yy = Mode(xx);

train_yy = Mode(Y_Train);

TrainingAccuracy = length(find(yy == train_yy))/size(train_yy,1);

disp(['Training Accuracy is : ', num2str(TrainingAccuracy * 100), ' %' ]);

tic;
%%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%


HH1 = [X_Test .1 * ones(size(X_Test,1),1)];

%clear test_x;

yy1=zeros(size(X_Test,1),N2*N1);

for i=1:N2

    beta1=beta11{i};ps1=ps(i);

    TT1 = HH1 * beta1;

    TT1  =  mapminmax('apply',TT1',ps1)';

clear beta1; clear ps1;

%yy1=[yy1 TT1];

yy1(:,N1*(i-1)+1:N1*i)=TT1;
end

clear TT1;

clear HH1;

HH2 = [yy1 .1 * ones(size(yy1,1),1)]; 

TT2 = tansig(HH2 * wh * l2);TT3=[yy1 TT2];

clear HH2;

clear wh;

clear TT2;

%%testing accuracy

x = TT3 * beta;

y = Mode(x);

test_yy = Mode(Y_Test);

TestingAccuracy = length(find(y == test_yy))/size(test_yy,1);

clear TT3;

Testing_time = toc;

disp('Testing has been finished!');

Accuracy=TestingAccuracy;