function[final_bias,final_var]=bias_var_d()

number_of_samples=100;

dataset=-1+(1+1)*rand(100,number_of_samples); %x_i
labels=2*(dataset).^2+guassian(dataset); %y_i=f(x_i)
true_y=2*(dataset).^2;

a=ones(number_of_samples,1);
lamda=[0.01 0.1 1 10];
final_bias=zeros(length(lamda),1);
final_var=zeros(length(lamda),1);
for l=1:length(lamda)
    for i=1:100
        x_=transpose(dataset(i,:));
        x=[a x_];
        x=[x x_.^2];
        y=transpose(labels(i,:));
        I=eye(size(x,2));
        w(i,:)=inv(transpose(x)*x+(lamda(l)*I))*transpose(x)*y;
    end
    w_0=w(:,1);
    w_1=w(:,2);
    w_2=w(:,3);
    prediction=zeros(100,number_of_samples);
    for i=1:100
        x=dataset(i,:);
        prediction(i,:)=w_0(i)+w_1(i)*x+w_2(i)*x.^2;
    end
    %estimate bias-var
    [final_bias(l),final_var(l)]=bi_var(prediction,true_y,number_of_samples);
end

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

function[value]=guassian(x)
exp_term=exp(-(x.^2)./(2*0.1));
value=1/(sqrt(2*pi*0.1)).*exp_term;