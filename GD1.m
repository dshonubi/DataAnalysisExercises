%Takes (x1,y1...xn,yn) as training data

% x = [(1:10);(1:10);(1:10)]; <- set x here 
% y = [0;1;0]; <- set y here
[o,p] = size(x);

% x = normr(x); normalizes the input x values for test data above would use 'zscore' for column otherwise 
weight= randn(1,p); % initialise the weights
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


