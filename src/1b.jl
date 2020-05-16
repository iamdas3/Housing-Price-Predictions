
using CSV
using Plots
using StatsBase

cd("/home/trideep/170010015/data")

dataset = CSV.read("HousingPrice_standardisedData.csv")

train = Int(floor(0.8*length(dataset.bedrooms_norm))) 
z1 = dataset.bedrooms_norm[1:train]

z2 = dataset.bathrooms_norm[1:train]

z3 = (dataset.bedrooms_norm[1:train]).^2

z4 = (dataset.bathrooms_norm[1:train]).^2

z5 = (dataset.bedrooms_norm[1:train]).*(dataset.bathrooms_norm[1:train])

m = length(z1)

x0 = ones(m)

X = cat(x0, z1, z2, z3, z4, z5, dims = 2)

price = dataset.price_simple[1:train]

Y = price

function costFunction(X,Y,B)
    m = length(Y)
    cost = sum(((X*B)-Y).^2)/(2*m)
    return cost
end

B = zeros(6,1)

function gradientDescent(X, Y, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    m = length(Y)
    for iteration in 1:numIterations
        loss = (X * B) - Y
        gradient = (X' * loss)/m
        B = B - learningRate * gradient
        cost = costFunction(X, Y, B)
        costHistory[iteration] = cost
    end
    return B, costHistory
end

learningRate = 0.001
newB, costHistory = gradientDescent(X, Y, B, learningRate, 10000)

YPred = X*newB

(((sum(YPred - Y)).^2)/(m)).^(0.5)

using DataFrames
Y_bar = zeros(17290)
for i = 1:17290
    Y_bar[i] = sum(Y/m)
end
1 - sum((YPred - Y).^2)/sum((Y - Y_bar).^2)

n = length(dataset.bedrooms_norm)
z1_test = dataset.bedrooms_norm[train:n]
z2_test = dataset.bathrooms_norm[train:n]
z3_test = (dataset.bedrooms_norm[train:n]).^2
z4_test = (dataset.bathrooms_norm[train:n]).^2
z5_test = (dataset.bedrooms_norm[train:n]).*(dataset.bathrooms_norm[train:n])
price_test = dataset.price_simple[train:n]

x0_test = ones(n+1-train)

X_test = cat(x0_test, z1_test, z2_test, z3_test, z4_test, z5_test, dims = 2)

Y_test = price_test

Y_Pred_test = X_test*newB
Y_Predicted = zeros(4324)
for i = 1:4324
    Y_Predicted[i] = Y_Pred_test[i]
end
Y_Predicted

(((sum(Y_Pred_test - Y_test)).^2)/(m)).^(0.5)

using DataFrames
Y_bar = zeros(4324)
for i = 1:4324
    Y_bar[i] = sum(Y/m)
end
1 - sum((Y_Pred_test - Y_test).^2)/sum((Y_test - Y_bar).^2)

print(newB)

using DataFrames
df = DataFrame(Price_predicted = Y_Predicted)

CSV.write("/home/trideep/170010015/data/1b.csv", df)


