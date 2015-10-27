disp('%%%%%%%% Results %%%%%%%%%%');

%%%%%%%%5 bias variance for 10 dataset %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

[g1,b1,v1,g2,b2,v2,g3,b3,v3,g4,b4,v4,g5,b5,v5,g6,b6,v6]=bias_variance_23(10);

disp('Dataset containing 10 samples each');
res_1=strcat('g1(x)--> Bias=',num2str(b1));
res1=strcat(res_1,'--> Variance=');
res1=strcat(res1,num2str(v1));
disp(res1);
res_1=strcat('g2(x)--> Bias=',num2str(b2));
res1=strcat(res_1,'--> Variance=');
res1=strcat(res1,num2str(v2));
disp(res1);
res_1=strcat('g3(x)--> Bias=',num2str(b3));
res1=strcat(res_1,'--> Variance=');
res1=strcat(res1,num2str(v3));
disp(res1);
res_1=strcat('g4(x)--> Bias=',num2str(b4));
res1=strcat(res_1,'--> Variance=');
res1=strcat(res1,num2str(v4));
disp(res1);
res_1=strcat('g5(x)--> Bias=',num2str(b5));
res1=strcat(res_1,'--> Variance=');
res1=strcat(res1,num2str(v5));
disp(res1);
res_1=strcat('g6(x)--> Bias=',num2str(b6));
res1=strcat(res_1,'--> Variance=');
res1=strcat(res1,num2str(v6));
disp(res1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% plots %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i=1;
subplot(4,3,i);
hist(g1,10);
i=i+1;
subplot(4,3,i);
hist(g2,10);
i=i+1;
subplot(4,3,i);
hist(g3,10);
i=i+1;
subplot(4,3,i);
hist(g4,10);
i=i+1;
subplot(4,3,i);
hist(g5,10);
i=i+1;
subplot(4,3,i);
hist(g6,10);
i=i+1;
disp('------------------------------')
[g1,b1,v1,g2,b2,v2,g3,b3,v3,g4,b4,v4,g5,b5,v5,g6,b6,v6]=bias_variance_23(100);
disp('Dataset containing 100 samples each');
res_1=strcat('g1(x)--> Bias=',num2str(b1));
res1=strcat(res_1,'--> Variance=');
res1=strcat(res1,num2str(v1));
disp(res1);
res_1=strcat('g2(x)--> Bias=',num2str(b2));
res1=strcat(res_1,'--> Variance=');
res1=strcat(res1,num2str(v2));
disp(res1);
res_1=strcat('g3(x)--> Bias=',num2str(b3));
res1=strcat(res_1,'--> Variance=');
res1=strcat(res1,num2str(v3));
disp(res1);
res_1=strcat('g4(x)--> Bias=',num2str(b4));
res1=strcat(res_1,'--> Variance=');
res1=strcat(res1,num2str(v4));
disp(res1);
res_1=strcat('g5(x)--> Bias=',num2str(b5));
res1=strcat(res_1,'--> Variance=');
res1=strcat(res1,num2str(v5));
disp(res1);
res_1=strcat('g6(x)--> Bias=',num2str(b6));
res1=strcat(res_1,'--> Variance=');
res1=strcat(res1,num2str(v6));
disp(res1);
subplot(4,3,i);
hist(g1,10);
i=i+1;
subplot(4,3,i);
hist(g2,10);
i=i+1;
subplot(4,3,i);
hist(g3,10);
i=i+1;
subplot(4,3,i);
hist(g4,10);
i=i+1;
subplot(4,3,i);
hist(g5,10);
i=i+1;
subplot(4,3,i);
hist(g6,10);
i=i+1;
disp('------------------------------')
lamda=[0.01 0.1 1 10];
[final_bias,final_var]=bias_var_d();
disp('With regularization');
for i=1:4
res_1=strcat('lambda =',num2str(lamda(i)));
res_11=strcat('--> Bias=',num2str(final_bias(i)));
res1=strcat(res_1,res_11);
res1=strcat(res1,'--> Variance=');
res1=strcat(res1,num2str(final_var(i)));
disp(res1);
end
%%%%%%%%%%  Question 6 %%%%%%%%%%%%%%%%%%%%%%%
[data,labels]=preprocess();
disp('Split=50-50');
disp('Iterations:3');
[avg_test_error,final_opt_lamda]=linear_ridge_regression(data,labels);
disp('Linear Ridge Regression');
disp('Average Test error');
disp(avg_test_error);
disp('Optimal lamda values');
disp(final_opt_lamda);
[avg_test_error,final_opt_lamda]=linear_kernel(data,labels);
disp('Linear Kernel');
disp('Average Test error');
disp(avg_test_error);
disp('Optimal lamda values');
disp(final_opt_lamda);
[avg_test_error,final_opt_lamda,final_opt_a,final_opt_b]=polynomial_kernel(data,labels);
disp('Polynomial kernel');
disp('Average Test error');
disp(avg_test_error);
disp('Optimal lamda values');
disp(final_opt_lamda);
disp('Optimal a values');
disp(final_opt_a);
disp('Optimal b values');
disp(final_opt_b);
[avg_test_error,final_opt_lamda,final_opt_a]=gaussian_kernel(data,labels);
disp('Gaussian kernel');
disp('Average Test error');
disp(avg_test_error);
disp('Optimal lamda values');
disp(final_opt_lamda);
disp('Optimal sigma^2 values');
disp(final_opt_a);