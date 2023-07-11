clc
clear

%% Load USIData

load('USIdata.mat')

Y_Train = dummyvar(Y_Train);

Y_Test = dummyvar(Y_Test);

%% BLS Training Parameter

C = 2^-30; s = .8;%the l2 regularization parameter and the shrinkage scale of the enhancement nodes

N11=10;%feature nodes  per group

N2=20;% number of group of feature nodes

N33=2000;% number of enhancement nodes

epochs=10;% number of epochs 

arf=0.8;


train_err=zeros(1,epochs);

test_err=zeros(1,epochs);

train_time=zeros(1,epochs);

test_time=zeros(1,epochs);

% Unbalance Modification
  
NicheCount = CalNicheCount(X_Train, Y_Train);

[C_X_Train,C_Y_Train]=UnbalanceModification(X_Train, Y_Train,NicheCount,arf);

%% DATA preprocessing
X_Train=[X_Train; C_X_Train];

Y_Train=[Y_Train; C_Y_Train];

% X_Train = normalize(X_Train, 'range');
% 
% X_Test = normalize(X_Test, 'range');

X_Train = zscore(X_Train')';

X_Test = zscore(X_Test')';

Y_Train=(Y_Train-1)*2+1;

Y_Test=(Y_Test-1)*2+1;

assert(isfloat(X_Train), 'Y_Train must be a float');
%% BLS Training

N1=N11; 

N3=N33;  

for j=1:epochs

    [Accuracy,Training_time] = oabls_train(X_Train,Y_Train,X_Test,Y_Test,s,C,N1,N2,N3);

    train_err(j)=(1-Accuracy); 
    
    train_time(j)=Training_time;

end
