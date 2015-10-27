function[avg_test_error,final_opt_lamda]=linear_ridge_regression(data,labels)

%calculate w
x0=ones(length(data),1);
data=[x0 data];
K=5;
lamda=[10^-4 10^-3 10^-2 10^-1 1 10 10^2 10^3];
error=zeros(3,1);
final_opt_lamda=zeros(3,1);
for t=1:3
    %random split 80-20
    size_dataset=length(data);
    split = round(size_dataset*0.8);
    rows = randperm(size_dataset);
    tr_data = data(rows(1:split),:);  %0.8 samples
    tr_label=labels(rows(1:split),:);
    te_data = data(rows(split+1:end),:); %0.2 samples
    te_label = labels(rows(split+1:end),:);
    
    %k-fold validation split
    [N,cols]=size(tr_data);
    Indices = crossvalind('Kfold', N, K);
    rows=unique(Indices);
    w=zeros(cols,length(lamda));
    optimal_lamda=zeros(length(rows),1);
    %Repeated 5 times
    for i=1:length(rows)
        x=tr_data(Indices(:)~=rows(i),:);
        y=tr_label(Indices(:)~=rows(i),:);
        validation_x=tr_data(Indices(:)==rows(i),:);
        validation_y=tr_label(Indices(:)==rows(i),:);
        I=eye(size(x,2));
        %pick optimal lambda
        error_valid=zeros(length(lamda),1);
        for l=1:length(lamda)
            w(:,l)=(x'*x+lamda(l)*I)\(x'*y);
            %Calculate mean squared error on validation set
            error_valid(l,1)=(sum((validation_y(:,1)-validation_x(:,:)*w(:,l)).^2))/length(validation_x);
        end
        %Report optimal lamda for this validation set
        [value,index]=min(error_valid);
        optimal_lamda(i)=lamda(index);
    end
    final_opt_lamda(t)=mode(optimal_lamda);
    %Train new model on tr_data
    model=(tr_data'*tr_data+final_opt_lamda(t,1)*I)\(tr_data'*tr_label);
    %Test error
    error(t)=(sum((te_label(:,1)-te_data(:,:)*model).^2))/length(te_data);
end
avg_test_error=sum(error(:))/3;
disp('lamda');
disp(final_opt_lamda);
disp('errors');
disp(error);
