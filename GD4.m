load 'Auto.data';
data = Auto; % loads Auto dataset - numbering,horsepower,weight,year,origin,high
data = data(:,(2:6)); % getting rid of numbering

rng(0428); %Same default generator as R == set.seed 

data = [data data(:,4)]; % reordering origin to make it easier to set dummy values
data = [data data(:,4)]; % calculate origin easier by multiplying the weight depending
data = [data data(:,4)]; % on origin value
data(:,4) = []; 

[e,r] = size(data);
for w = 1:e
    
if data(w,5) == 1
    
   data(w,5) = 0;
   data(w,6) = 0;
   data(w,7) = 0;
else
   if data(w,6) == 2
    
   data(w,5) = 0;
   data(w,6) = 1;
   data(w,7) = 0; 
   
   else
      if data(w,7) == 3
    
      data(w,5) = 0;
      data(w,6) = 0;
      data(w,7) = 1; 
      end
   end
end
end   
%set values that are equal to train or test to the dataset which you want to use
perm = randperm(392);
train = data(perm(1:196),:);  % initialising training set
test = setdiff(data,train,'rows'); % initialising test set

y = train(:,4); % observed data
train(:,4) = []; %separates observed data from input values <- essentially (xi1..xij,yi)

x=train;
x(:,(1:3)) = normalize(train(:,(1:3)),'zscore');%normalizes the input x values

[o,p] = size(train);
weight= -0.7 + (0.7--0.7).*rand(1,p); % initialised the weights between -0.7 and 0.7
bias = 0; % initialise the bias
learningRate =0.1; %set learning rate
count =0;
MSEUpdate = [];

while (count<1000) 
  
weightX=x.*weight; %B1X1(1) + B2X1(2)+ ... +BiXi(j)
[m,n] = size(weightX);
Y = zeros(m,1);
stepSize = zeros(m,1);

for i = 1:m
    biasX = sum(weightX(i,:)) + bias; % B0 + to each row of weights
    biasArray = dlarray(biasX,'B');
    Y(i,:)= sigmoid(biasArray); %predicted value after sigmoid activation function
end

minError = y - Y; % observed - predicted
minError = minError.*minError; %squared
MSE=sum(minError)/length(minError); % Mean Squared Error

MSEUpdate=[MSEUpdate MSE]; %Checks MSE is being reduce each iteration of the loop for minimisation

coef = 2.*(y-Y).*(Y.*(1-Y)); %derivative of sigma function

%Updating weights for B1...Bj
for a = 1:p 
     weight(1,a)= weight(1,a)+(learningRate/o).*(sum(coef.*x(:,a)));  
end 

bias = bias+(learningRate/o).*(sum(coef)); %BO - xi0 =1 1*coef = coef
count = count +1;

end

%Used some expensive functions, so may take time to run
