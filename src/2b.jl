
using CSV
using StatsBase

cd("/home/trideep/170010015/data")

dataset = CSV.read("HousingPrice_normalisedData_lasso.csv")

train = Int(floor(0.6*length(dataset.bedrooms_normed)))
bedrooms = dataset.bedrooms_normed[1:train]

bathrooms = dataset.bathrooms_normed[1:train]

sqft_living = dataset.sqft_living_normed[1:train]

price = dataset.price_normed[1:train]

X = (cat(bedrooms, bathrooms, sqft_living, dims = 2))
m,n = size(X)
maximum(X[:,2])

Y = price

function costFunction(X,Y,B,alpha)
    m = length(Y)
    cost = sum((Y-(X*B)).^2) + alpha*sum((abs.(B)))
    return cost
end


function soft_threshold(rho, alpha)
    if rho < (-alpha/2)
        return (rho +(alpha/2))
    elseif rho > (alpha/2)
        return (rho-(alpha/2))
    else
        return 0
    end
end

function coordinate_descent_lasso(B, X, Y, alpha, num_iters)
    for i in 1:num_iters
        for j in 1:n
	    B[j] = 1
            rho = sum((transpose(X[:,j])*(Y- (X*B) + B[j]*X[:,j])))
            B[j] = soft_threshold(rho, alpha)
        end
    end
    return B
end

B_0 = [1.0,1.0,1.0]

B_var = coordinate_descent_lasso(B_0, X, Y, 100, 100)

Y_Pred = abs.(X*B_var)

Y - Y_Pred


