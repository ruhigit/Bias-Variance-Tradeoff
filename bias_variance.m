%Bias/Variance Tradeoff
%f(x)=2x^2+E
function[g1,g2]=bias_variance()
% Generate 100 datasets, each containing 10 samples(x,y)
dataset=-1+(1+1)*rand(100,10); %x_i
labels=2*dataset+guassian(dataset); %y_i=f(x_i)
%for each of the 6 g functions

%g1(x)=1
%Estimate its parameters using linear regression
%Here no parameters
%Hence Compute the sum-square-error on every dataset
loss=zeros(100,1);
for i=1:100
loss(i,1)=sum((1-labels(i,:)).^2);
end
g1=loss;

%g2(x)=w0
a=ones(10,1);
w_0=estimate_par(a,labels);
%compute the sum-square-error on every dataset
for i=1:100
    loss(i,1)=sum((labels(i,:)-w_0).^2);
end
g2=loss;

%g3(x)=w0+w1x
a=ones(10,1);
w=estimate_par(x,labels);
%compute the sum-square-error on every dataset
for i=1:100
    loss(i,1)=sum((labels(i,:)-w_0).^2);
end
g2=loss;


function[w]=estimate_par(x,labels)
%Estimate its parameters using linear regression
for i=1:100
    y=transpose(labels(i,:));
    w=inv(transpose(x)*x)*transpose(x)*y;
end
function[value]=guassian(x)
exp_term=exp(-(x.^2)./(2*0.1));
value=1/(sqrt(2*pi*0.1)).*exp_term;
