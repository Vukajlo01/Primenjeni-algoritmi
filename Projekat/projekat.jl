# Pkg
using Statistics
using StatsBase
using DataFrames
using CSV
using Flux
using Flux
using Lathe


# DataFrameCSV
df = DataFrame(CSV.File("winequality-red.csv"))
dftrain, dftest = Lathe.preprocess.TrainTestSplit(df, .80)

#Model
model = Dense(11, 1)

train_input = convert(Array, select(dftrain, Not(:quality)))'
train_output = convert(Array, select(dftrain, :quality))'

loss(x, y) = @show Flux.mse(model(x), y)

#Training
for i in 1:1000
    Flux.train!(loss, params(model), [(train_input, train_output)], Descent(0.0001))
end

params(model)

#Testing
test_input = convert.(Array, eachrow(select(dftest, Not(:quality))))
test_output = convert(Array, select(dftest, :quality))

predictions = vcat(model.(test_input)...)

#greske
errors = test_output .- predictions

# Porsecna apsolutna greska (MAE)
meanAbsError = mean(abs.(errors))
@show meanAbsError
# Prosecna relativna greska (MAPE)
meanRelError = mean(abs.(errors./dftest.quality))
@show meanRelError
# Prosek kvadrata GRESKE (MSE)
meanErrorSquared = mean(errors.*errors)
@show meanErrorSquared
# Koren proseka kvadrata gresaka (RMSE)
rootMeanErrorSquared = sqrt(meanErrorSquared)
@show rootMeanErrorSquared