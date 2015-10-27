function[data,labels]=preprocess()

%Reference from:http://www.mathworks.com/help/matlab/ref/fscanf.html
fileID=fopen('data.txt','r');
formatSpec='%e %e %e %e %e %e %e';
sizeA=[7 Inf];
matrix=fscanf(fileID,formatSpec,sizeA);
fclose(fileID);
matrix=matrix';


data=matrix(:,2:7);
labels=matrix(:,1);

%%Normalize data
no_data=length(data);
for i=1:6
    avg(i)=sum(data(:,i))/no_data;
    std_dev(i)= (sum((data(:,i)-avg(i)).^2))/(no_data-1);
    data(:,i)=(data(:,i)-avg(i))/sqrt(std_dev(i));
   
end