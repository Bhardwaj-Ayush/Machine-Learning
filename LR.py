from sklearn import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrices import mean_squared_error

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y)

model = LinearRegression()

model.fits(x_train,y_train)

pred = model.predict(x_test)

mse = mean_squared_error(y_test,pred)

print(mse)