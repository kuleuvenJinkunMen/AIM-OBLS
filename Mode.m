function y=Mode(x)

for i=1:size(x,1)
[~,y(i)]=max(x(i,:));
end
y=y';
end
