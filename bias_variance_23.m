%Bias/Variance Tradeoff
%f(x)=2x^2+E
function[g1,b1,v1,g2,b2,v2,g3,b3,v3,g4,b4,v4,g5,b5,v5,g6,b6,v6]=bias_variance_23(number_of_samples)
% Generate 100 datasets, each containing number_of_samples samples(x,y)
dataset=-1+(1+1)*rand(100,number_of_samples); %x_i
labels=2*(dataset).^2+guassian(dataset); %y_i=f(x_i)
true_y=2*(dataset).^2;
%for each of the 6 g functions

[g1,b1,v1]=g1_func(dataset,labels,number_of_samples,true_y);
[g2,b2,v2]=g2_func(dataset,labels,number_of_samples,true_y);
[g3,b3,v3]=g3_func(dataset,labels,number_of_samples,true_y);
[g4,b4,v4]=g4_func(dataset,labels,number_of_samples,true_y);
[g5,b5,v5]=g5_func(dataset,labels,number_of_samples,true_y);
[g6,b6,v6]=g6_func(dataset,labels,number_of_samples,true_y);

function[loss,bias,variance]=g1_func(dataset,labels,number_of_samples,true_y)
%g1(x)=1
%Estimate its parameters using linear regression
%Here no parameters
loss=zeros(100,1);
%Hence Compute the sum-square-error on every dataset
prediction=ones(100,number_of_samples);
for i=1:100
loss(i,1)=(sum((1-labels(i,:)).^2))/number_of_samples;
end

%bias^2
[bias,variance]=bi_var(prediction,true_y,number_of_samples);

function[loss,bias,variance]=g2_func(dataset,labels,number_of_samples,true_y)
%g2(x)=w0
a=ones(number_of_samples,1);
x=a;
for i=1:100
    y=transpose(labels(i,:));
    w(i)=(transpose(x)*x)\(transpose(x)*y);
end
w_0=w;
%compute the sum-square-error on every dataset
loss=zeros(100,1);
prediction=ones(100,number_of_samples);
for i=1:100
    prediction(i,:)=w_0(i);
    loss(i,1)=(sum((labels(i,:)-w_0(i)).^2))/number_of_samples;
end

err_b=zeros(100,1);
err_v=zeros(100,1);
for i=1:100
    efx=w_0(i);
    err_b(i)=((prediction(i,:)-true_y(i,:))*(prediction(i,:)-true_y(i,:))')/number_of_samples;
    err_v(i)=sum((prediction(i,:)-efx).^2)/number_of_samples;
   
end
bias=sum(err_b)/(100);
variance=sum(err_v)/(100);

function[loss,bias,variance]=g3_func(dataset,labels,number_of_samples,true_y)
%g3(x)=w0+w1x
a=ones(number_of_samples,1);
for i=1:100
    x=transpose(dataset(i,:));
    x=[a x];
    y=transpose(labels(i,:));
    w(i,:)=(transpose(x)*x)\(transpose(x)*y);
end
w_0=w(:,1);
w_1=w(:,2);
prediction=zeros(100,number_of_samples);
%compute the sum-square-error on every dataset
for i=1:100
    prediction(i,:)=(w_0(i)+w_1(i)*dataset(i,:));
    loss(i,1)=(sum((labels(i,:)-(prediction(i,:))).^2))/number_of_samples;
end
%bias^2
[bias,variance]=bi_var(prediction,true_y,number_of_samples);

function[loss,bias,variance]=g4_func(dataset,labels,number_of_samples,true_y)
%g4(x)=w0+w1x+w2x^2
a=ones(number_of_samples,1);
for i=1:100
    x_=transpose(dataset(i,:));
    x=[a x_];
    x=[x x_.^2];
    y=transpose(labels(i,:));
    w(i,:)=inv(transpose(x)*x)*transpose(x)*y;
end
w_0=w(:,1);
w_1=w(:,2);
w_2=w(:,3);
%compute the sum-square-error on every dataset
prediction=zeros(100,number_of_samples);
for i=1:100
    x=dataset(i,:);
    prediction(i,:)=w_0(i)+w_1(i)*x+w_2(i)*x.^2;
    loss(i,1)=(sum((labels(i,:)-(prediction(i,:))).^2));
end

[bias,variance]=bi_var(prediction,true_y,number_of_samples);

function[loss,bias,variance]=g5_func(dataset,labels,number_of_samples,true_y)
%g5(x)=w0+w1x+w2x^2+w3x^3
a=ones(number_of_samples,1);
for i=1:100
    x_=transpose(dataset(i,:));
    x=[a x_];
    x=[x x_.^2];
    x=[x x_.^3];
    y=transpose(labels(i,:));
    w(i,:)=(transpose(x)*x)\(transpose(x)*y);
end
w_0=w(:,1);
w_1=w(:,2);
w_2=w(:,3);
w_3=w(:,4);
%compute the sum-square-error on every dataset
prediction=zeros(100,number_of_samples);
for i=1:100
    x=dataset(i,:);
    prediction(i,:)=w_0(i)+w_1(i)*x+w_2(i)*x.^2+w_3(i)*x.^3;
    loss(i,1)=(sum((labels(i,:)-(prediction(i,:))).^2))/number_of_samples;
end

[bias,variance]=bi_var(prediction,true_y,number_of_samples);

function[loss,bias,variance]=g6_func(dataset,labels,number_of_samples,true_y)
%g6(x)=w0+w1x+w2x^2+w3x^3+w4x^4
a=ones(number_of_samples,1);
for i=1:100
    x_=transpose(dataset(i,:));
    x=[a x_];
    x=[x x_.^2];
    x=[x x_.^3];
    x=[x x_.^4];
    y=transpose(labels(i,:));
    w(i,:)=inv(transpose(x)*x)*transpose(x)*y;
end
w_0=w(:,1);
w_1=w(:,2);
w_2=w(:,3);
w_3=w(:,4);
w_4=w(:,5);
%compute the sum-square-error on every dataset
prediction=ones(100,number_of_samples);
for i=1:100
    x=dataset(i,:);
    prediction(i,:)=w_0(i)+w_1(i)*x+w_2(i)*x.^2+w_3(i)*x.^3+w_4(i)*x.^4;
    loss(i,1)=(sum((labels(i,:)-(prediction(i,:))).^2))/number_of_samples;
end

%bias^2
[bias,variance]=bi_var(prediction,true_y,number_of_samples);

function[value]=guassian(x)
exp_term=exp(-(x.^2)./(2*0.1));
value=1/(sqrt(2*pi*0.1)).*exp_term;

function[b,v]=bi_var(prediction,true_y,number_of_samples)
err_b=zeros(100,1);
err_v=zeros(100,1);
for i=1:100
    efx=(sum(prediction(i,:)))/number_of_samples;
    err_b(i)=((prediction(i,:)-true_y(i,:))*(prediction(i,:)-true_y(i,:))')/number_of_samples;
    err_v(i)=sum((prediction(i,:)-efx).^2)/number_of_samples;
   
end
b=sum(err_b)/(100);
v=sum(err_v)/(100);
