# The libs are needed for plotting the data, 
# and for the sklearn model only.
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score
import numpy as np

x_data = [1.0, 2.0, 4.3, 3.1, 5.2, 6.6]
y_data = [2.5, 3.1, 3.9, 4.2, 5.3, 7.1]
# x_test_data = [1.0, 4.0, 5.14, 3.2, 9.5, 4.65]
# needed to test the sklearn model
x_test_data = [1.0, 2.0, 4.3, 3.1, 5.2, 6.6]
y_test_data = [2.8, 4.0, 4.5, 5.0, 4.7, 6.8]

# The MSE cost funtion
def costFun(xData, yData, wt, bias):
    # MSE
    N = len(xData)
    tError = 0
    for i in range(N):
        # 1/N summation(y - (mx + b))^2
        tError += (yData[i] - (wt * xData[i] + bias)) ** 2
    return tError/N

# Gradient descent function
def gdUpdateWt(xData, yData, wt, bias, LR):
    wtDv = 0
    bsDv = 0
    N = len(xData)

    for i in range(N):
        #gradient (derivatives) of the cost func wrt to wt & bias.
        # -2x(y - (mx + b))
        wtDv += -2 * (xData[i] * (yData[i] - (wt * xData[i] + bias) ))

        # -2(y - (mx + b))
        bsDv += -2 * (yData[i] - (wt * xData[i] + bias) )

    # subtracting because gradient "DESCENT"!
    wt -= (wtDv/N) * LR
    bias -= (bsDv/N) * LR

    return wt, bias

# training- basically run the above functions K times and log data.
def trainM(xData, yData, wt, bias, LR, iters):
    costLog = []
    ranIters = []

    for i in range(iters):
        wt,bias = gdUpdateWt(xData, yData, wt, bias, LR)

        cost = costFun(xData, yData, wt, bias)
        costLog.append(cost)

        # if i%10 == 0:
        print(f"iter={i}, wt={wt}, bias={bias}, cost={cost}")
            # costLog.append(cost)
        ranIters.append(i)

    
    return wt,bias,costLog,ranIters

finalWt, finalBias, finalCostLog, itersRan = trainM(x_data, y_data, 0.0, 0.0, 0.03, 1000)
print(f"finalWt: {finalWt}")
print(f"finalBias: {finalBias}")
# print(f"costLog: {finalCostLog}")

estYData = []
for xd in x_data:
    # prediction = y[i] = wt * x[i] + b
    estYData.append(finalWt*xd + finalBias)


# plotting 
plt.scatter(x_data, y_data, c="green", label="original data")
plt.plot(x_data, estYData, c="red")
plt.scatter(x_data, estYData, c="red", label="pred. data")
plt.title("pred")
plt.xlabel("x-values (independent)")
plt.ylabel("y-values (dependent)")
plt.legend()

# plt.scatter(x_data, y_data, c="green", label="original data")
# plt.plot(x_data, estYData, c="red")
# plt.scatter(x_data, estYData, c="red", label="pred. data")
# plt.title("original data")
# plt.xlabel("x-values (independent)")
# plt.ylabel("y-values (dependent)")
# plt.legend()

# plt.savefig("final result_mym.png")

# plotting for 10th iter.
# plt.scatter(x_data, y_data, c="green", label="original data")
# plt.plot(x_data, estYData, c="red")
# plt.scatter(x_data, estYData, c="red", label="pred. data")
# plt.title("final")
# plt.xlabel("x-values (independent)")
# plt.ylabel("y-values (dependent)")
# plt.legend()
# plt.savefig("final result.png")

# plt.plot(finalCostLog, itersRan, c="red")
# plt.title("Error rate")
# plt.ylabel("iterations ran")
# plt.xlabel("(MSE) cost fun")
# plt.savefig("error-plot.png")




# cross-check using sklearn LR model.

x_train = np.array(x_data)
y_train = np.array(y_data)
x_test = np.array(x_test_data)
y_test = np.array(y_test_data)

x_train = x_train.reshape(-1,1)
x_test = x_test.reshape(-1,1)

clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)

print("R^2 from sklearn LR: ")
print(r2_score(y_test,y_pred))

print("R^2 from this implementation:")
print(r2_score(y_test, estYData))

# plt.scatter(x_data, y_data, c="green", label="original data")
# plt.plot(x_data, y_pred, c="red")
# plt.scatter(x_data, estYData, c="red", label="pred. data")
# plt.title("pred. by sklearn l-reg model")
# plt.xlabel("x-values (independent)")
# plt.ylabel("y-values (dependent)")
# plt.legend()
# plt.savefig("final result_sklearn.png")

plt.show()

