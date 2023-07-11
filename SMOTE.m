function [C_X,C_Y]=SMOTE(Nx,X,Y,arf)

% SMOTE

[m,~] = size(Nx);

randomIndex = randi([1, m]);

Xneo=Nx(randomIndex,:);


C_X = X + arf*( X - Xneo);

C_Y = Y;

end