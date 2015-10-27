function[avg_test_error,final_opt_lamda]=linear_kernel_backup(da,la)

data=da(1:500,:);
labels=la(1:500,:);

%kernel function is
kernel_fun=@(x_i,x_j) x_i*x_j';

%append column of 1
x0=ones(length(data),1);
data=[x0 data];

%range of values of lamda
lamda=[10^-4 10^-3 10^-2 10^-1 1 10 10^2 10^3];

%5-fold cross validation
K_crossvalid=5;

%final
error=zeros(10,1);
final_opt_lamda=zeros(10,1);

%run for 10 iterations
for t=1:10
   
    %random split 80-20
    size_dataset=length(data);
    split = round(size_dataset*0.8);
    rows = randperm(size_dataset);
    tr_data = data(rows(1:split),:);  %0.8 samples
    tr_label=labels(rows(1:split),:);
    te_data = data(rows(split+1:end),:); %0.2 samples
    te_label = labels(rows(split+1:end),:);
    
    %k-fold validation split
    N=length(tr_data);
    Indices = crossvalind('Kfold', N, K_crossvalid);
    rows=unique(Indices);
    optimal_lamda=zeros(length(rows),1);
    
    %Repeated 5 times
    for i=1:length(rows)
        x=tr_data(Indices(:)~=rows(i),:);
        y=tr_label(Indices(:)~=rows(i),:);
        validation_x=tr_data(Indices(:)==rows(i),:);
        validation_y=tr_label(Indices(:)==rows(i),:);
        size_data=length(x);
        %%%
        %%%Compute K matrix
        K=zeros(size_data,size_data);
        for M=1:size_data
            for N=1:size_data
                K(M,N)=kernel_fun(x(M,:),x(N,:));
            end
        end
        I=eye(length(K));
        %pick optimal lambda
        error_valid=zeros(length(lamda),1);
        len_valid=length(validation_x);
        y_hat=zeros(len_valid,1);
   %%% for each value of lamba calculate error
        for l=1:length(lamda)
            for va=1:length(validation_x)
                i1=1:1:size_data;
                k_small(:,1)=(kernel_fun(x(i1,:),validation_x(va,:)));
            y_hat(va,1)=y'*((K+lamda(l)*I)\ k_small);
            end
          
            %Calculate mean squared error on validation set
            error_valid(l,1)=(sum((validation_y(:,1)-y_hat(:,1)).^2))/length(validation_x);
        end
        %Report optimal lamda for this validation set
        [value,index]=min(error_valid);
        optimal_lamda(i)=lamda(index);
    end
    final_opt_lamda(t)=mode(optimal_lamda);
    %Train new model on tr_data
    size_trdata=length(tr_data);
    I=eye(size_trdata);
     K_=zeros(size_trdata,size_trdata);
        for M=1:size_trdata
            for N=1:size_trdata
                K_(M,N)=kernel_fun(tr_data(M,:),tr_data(N,:));
            end
        end
        y_hat_=zeros(length(te_data),1);
        
        for va=1:length(te_data)
            i1=1:1:size_trdata;
            k_small_(:,1)=(kernel_fun(tr_data(i1,:),te_data(va,:)));
            y_hat_(va,1)=tr_label'*((K_+final_opt_lamda(t)*I)\k_small_);
         end
    %Test error
    error(t)=(sum((te_label(:,1)-y_hat_(:,1)).^2))/length(te_data);
end
avg_test_error=sum(error(:))/10;
disp('lamda');
disp(final_opt_lamda);
disp('errors');
disp(error);
