function [C_X_Train,C_Y_Train]=UnbalanceModification(X_Train, Y_Train,NicheCount,arf)
%Unbalanced Data Modification
isInf = isinf(NicheCount);

nonNicheCount = NicheCount(~isInf);

u=mean(nonNicheCount);

N=length(NicheCount);

C_X_Train=[];

C_Y_Train=[];

for n=1:N
    
    L=find(Y_Train(n,:)==1);
    
    Nx = X_Train(find(Y_Train(:,L)==1),:);

    X=X_Train(n,:);

    Y=Y_Train(n,:);

    a = 1;

    while NicheCount(n)<u*0.7

          [C_X,C_Y]=SMOTE(Nx,X,Y,arf);

          C_X_Train=[C_X_Train;C_X];

          C_Y_Train=[C_Y_Train;C_Y];

          a=a+1;

          if a>=3
              break;
          end

          NicheCount = CalNicheCount([X_Train;C_X_Train], [Y_Train;C_Y_Train]);

    end
end

    
   
end