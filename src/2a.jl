
using CSV
using Plots
using StatsBase
using LinearAlgebra
using DataFrames

cd("/home/trideep/170010015/data")

dataset = CSV.read("HousingPrice_normalisedData.csv")

train = Int(floor(0.6*length(dataset.bedrooms_normed)))
bedrooms = dataset.bedrooms_normed[1:train]

bathrooms = dataset.bathrooms_normed[1:train]

sqft_living = dataset.sqft_living_normed[1:train]

m = length(bedrooms)

x0 = ones(m)

X_test = cat(x0, bedrooms, bathrooms, sqft_living, dims = 2)

price = dataset.price_simple[1:train]
Y_test = price

m,n = size(X_test)
Idn =1.0* Matrix(I, n, n)

function costFunction(X,Y,B)
    m = length(Y)
    cost = sum(((X*B)-Y).^2)/(2*m) +sum(B.^2)
    return cost
end

function Ridge_regression(X,Y,alpha)
    B = inv((transpose(X)*X) + alpha*Idn)*(transpose(X))*Y
    return B
end

K = zeros(m)
for i in 1:100
    Ridge_regression(X_test,Y_test,i)
    YPred = X_test*B
    K[i] = (((sum(YPred - Y_test)).^2)/(m)).^(0.5)
end

K

validation = Int(floor(0.8*length(dataset.bedrooms_normed)))
bedrooms = dataset.bedrooms_normed[train:validation]

bathrooms = dataset.bathrooms_normed[train:validation]
l = length(bathrooms)
x0_validation = ones(l)

sqft_living = dataset.sqft_living_normed[train:validation]

price = dataset.price_simple[train:validation]

X_validation = cat(x0_validation, bedrooms, bathrooms, sqft_living, dims = 2)

Y_validation = price

B_validation = Ridge_regression(X_validation,Y_validation,1)
Y_Pred = X_validation*B_validation
(((sum(Y_Pred - Y_validation)).^2)/(m)).^(0.5)

L = zeros(m)
for i in 1:100
    B_validation = Ridge_regression(X_validation,Y_validation,i)
    Y_Pred = X_validation*B_validation
    L[i] = (((sum(Y_Pred - Y_validation)).^2)/(m)).^(0.5)
end

L

test = Int(floor(1*length(dataset.bedrooms_normed)))
bedrooms = dataset.bedrooms_normed[validation:test]
bathrooms = dataset.bathrooms_normed[validation:test]
sqft_living = dataset.sqft_living_normed[validation:test]
price = dataset.price_simple[validation:test]

n = length(bedrooms)
x0_test = ones(n)
X_test = cat(x0_test, bedrooms, bathrooms, sqft_living, dims = 2)
Y_test = price

Y_Predicted = X_test*B_validation
RMSE = (((sum(Y_Predicted - Y_test)).^2)/(m)).^(0.5)

Y_bar = zeros(4324)
for i = 1:4324
    Y_bar[i] = sum(Y_test/m)
end

R_squared = 1 - sum((Y_Pred - Y_test).^2)/sum((Y_test - Y_bar).^2)

print(B_validation)

df = DataFrame(Predicted_price = Y_Predicted) 

CSV.write("/home/trideep/170010015/data/2a.csv", df)


