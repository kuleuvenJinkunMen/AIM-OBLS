function NicheCount = CalNicheCount(X_Train, Y_Train)
%Niche Count Calculation 

[Ms,nObj]=size(X_Train);

[~,Nl]=size(Y_Train);

NicheCount=zeros(1,Ms);

for k = 1:Nl
        
        s=find(Y_Train(:,k)==1);

        TempX=X_Train(s,:)';
 
        n= length(s);
        
        d = zeros(n, nObj);
        
        for j = 1:nObj
            
            [cj, so] = sort(TempX(j, :));
            
            d(so(1), j) = inf;
            
            for i = 2:n-1
                
                d(so(i), j) = abs(cj(i+1)-cj(i-1))/abs(cj(1)-cj(end));
                
            end
            
            d(so(end), j) = inf;
            
        end
        
        
        for i = 1:n
            
            NicheCount(s(i)) = sum(d(i, :));
            
        end
        
 end






end