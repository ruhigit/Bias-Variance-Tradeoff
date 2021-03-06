function[avg_test_error,final_opt_lamda,final_opt_a]=gaussian_kernel(da,la)

data=da;
labels=la;
%kernel function is
kernel_fun=@(x_i,x_j,a) exp(-((x_i(1,:)-x_j(1,:))*(x_i(1,:)-x_j(1,:))'/a));
%kernel_fun=@(x_i,x_j,a) exp(-(norm(x_i(1,:)-x_j(1,:)))/a);

%append column of 1
x0=ones(length(data),1);
data=[x0 data];

%range of values of lamda
lamda=[10^-4 10^-3 10^-2 10^-1 1 10 10^2 10^3];
a=[0.125 0.25 0.5 1 2 4 8];
%5-fold cross validation
K_crossvalid=5;

%final
error=zeros(3,1);
final_opt_lamda=zeros(3,1);
final_opt_a=zeros(3,1);


%run for 10 iterations
for t=1:3
   
    disp('Splitting data in 50-50 for train and test');
    %random split 50-50
    size_dataset=length(data);
    split = round(size_dataset*0.5);
    rows = randperm(size_dataset);
    tr_data = data(rows(1:split),:);  %0.5 samples
    tr_label=labels(rows(1:split),:);
    te_data = data(rows(split+1:end),:); %0.5 samples
    te_label = labels(rows(split+1:end),:);
    
    disp('Executing 5-fold validation');
    
    %k-fold validation split
    N=length(tr_data);
    Indices = crossvalind('Kfold', N, K_crossvalid);
    rows=unique(Indices);
    
    %initialize
    optimal_lamda=zeros(length(rows),1);
    optimal_a=zeros(length(rows),1);
    optimal_err=zeros(length(rows),1);
    
    disp('For each set');
    %Repeated 5 times
    for i=1:length(rows)
        x=tr_data(Indices(:)~=rows(i),:);
        y=tr_label(Indices(:)~=rows(i),:);
        validation_x=tr_data(Indices(:)==rows(i),:);
        validation_y=tr_label(Indices(:)==rows(i),:);
        size_data=length(x);
        %%%
        %pick optimal lambda
        size_par=length(lamda)*length(a);
        %error_valid will be 180*3 matrix
        %lamda a  error
        error_valid=zeros(size_par,3);
        len_valid=length(validation_x);
        y_hat=zeros(len_valid,1);
   %%% for each value of lamba calculate error
        err=1;
        for l=1:length(lamda)  
            for a_1=1:length(a)
                    error_valid(err,1)=lamda(l);
                    error_valid(err,2)=a(a_1);
                  
                    %%%Compute K matrix
                    K=zeros(size_data,size_data);
                    for M=1:size_data
                        for N=1:size_data
                            K(M,N)=kernel_fun(x(M,:),x(N,:),a(a_1));
                        end
                    end
                    I=eye(length(K));
                    k_small=zeros(size_data,1);
                    for va=1:length(validation_x)
                        i1=1:1:size_data;
                        k_small(i1,1)=kernel_fun(x(i1,:),validation_x(va,:),a(a_1));
                        y_hat(va,1)=y'*((K+lamda(l)*I)\ k_small);
                    end
                              %Calculate mean squared error on validation set
                    error_valid(err,3)=(sum((validation_y(:,1)-y_hat(:,1)).^2))/length(validation_x);
                    err=err+1;
            end
        end
        %Report optimal lamda for this validation set
        [value,index]=min(error_valid(:,3));
        optimal_lamda(i)=error_valid(index,1);
        optimal_a(i)=error_valid(index,2);
        optimal_err(i)=value;
    end
    [fin,id]=min(optimal_err);
    final_opt_lamda(t)=(optimal_lamda(id));
    final_opt_a(t)=(optimal_a(id));
   
    %Train new model on tr_data
    size_trdata=length(tr_data);
    I=eye(size_trdata);
     K_=zeros(size_trdata,size_trdata);
        for M=1:size_trdata
            for N=1:size_trdata
                K_(M,N)=kernel_fun(tr_data(M,:),tr_data(N,:),final_opt_a(t));
            end
        end
        y_hat_=zeros(length(te_data),1);
        k_small_=zeros(size_trdata,1);
        for va=1:length(te_data)
            i1=1:1:size_trdata;
            k_small_(i1,1)=(kernel_fun(tr_data(i1,:),te_data(va,:),final_opt_a(t)));
            y_hat_(va,1)=tr_label'*((K_+final_opt_lamda(t)*I)\k_small_);
         end
    %Test error
    error(t)=(sum((te_label(:,1)-y_hat_(:,1)).^2))/length(te_data);
end
avg_test_error=sum(error(:))/3;
avg_test_error=avg_test_error/10;
final_opt_lamda=final_opt_lamda/1000;
