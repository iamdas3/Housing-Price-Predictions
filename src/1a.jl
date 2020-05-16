
using CSV
using Plots
using StatsBase

cd("/home/trideep/170010015/data")

dataset = CSV.read("HousingPrice_standardisedData.csv")

train = Int(floor(0.8*length(dataset.bedrooms_norm)))
bedrooms = dataset.bedrooms_norm[1:train]

bathrooms = dataset.bathrooms_norm[1:train]

sqft_living = dataset.sqft_living_norm[1:train]

price = dataset.price_simple[1:train]

m = length(bedrooms)

x0 = ones(m)

X = cat(x0, bedrooms, bathrooms, sqft_living, dims = 2)

Y = price

function costFunction(X,Y,B)
    m = length(Y)
    cost = sum(((X*B)-Y).^2)/(2*m)
    return cost
end

B = zeros(4,1)

function gradientDescent(X, Y, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    m = length(Y)
    for iteration in 1:numIterations
        loss = (X*B) - Y
        gradient = (X' * loss)/m
        B = B - learningRate*gradient
        cost = costFunction(X, Y, B)
        costHistory[iteration] = cost
    end
    return B, costHistory
end

learningRate = 0.01
newB, costHistory = gradientDescent(X, Y, B, learningRate, 1000000)

YPred = abs.(X*newB)

(((sum(YPred - Y)).^2)/(m)).^(0.5)

using DataFrames
Y_bar = zeros(17290)
for i = 1:17290
    Y_bar[i] = sum(Y/m)
end
1 - sum((YPred - Y).^2)/sum((Y - Y_bar).^2)

n = length(dataset.bedrooms_norm)
test_bedrooms = dataset.bedrooms_norm[train:n]
print(test_bedrooms)
test_bathrooms = dataset.bathrooms_norm[train:n]
print(test_bathrooms)
test_sqft_living = dataset.sqft_living_norm[train:n]
print(test_sqft_living)
test_price = dataset.price_simple[train:n]
print(test_price)

x0_test = ones(n-train+1)

X_test = cat(x0_test, test_bedrooms, test_bathrooms, test_sqft_living, dims = 2)

Y_test = test_price

print(newB)
Y_Pred = X_test*newB
length(Y_Pred)
Y_Predicted = zeros(4324)
for i = 1:4324
    Y_Predicted[i] = Y_Pred[i]
end
Y_Predicted

(((sum(Y_Pred - Y_test)).^2)/(m)).^(0.5)

using DataFrames
Y_bar = zeros(4324)
for i = 1:4324
    Y_bar[i] = sum(Y/m)
end
1 - sum((Y_Pred - Y_test).^2)/sum((Y_test - Y_bar).^2)

df = DataFrame(Price_predicted = Y_Predicted)

CSV.write("/home/trideep/170010015/data/1a.csv", df)




