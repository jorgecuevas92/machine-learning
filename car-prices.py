import numpy as np
import matplotlib.pyplot as pyplot

x = np.array([
    2011,
    2013,
    2001,
    2016,
    2012,
    2017,
    2011,
    2014,
    2008,
    2012,
    2015,
    2017])

y = np.array([
    10500,
    13500,
    6000,
    23000,
    14000,
    25000,
    17500,
    18000,
    11500,
    13000,
    17500,
    24000])

test = np.array([
    [2011],
    [2013],
    [2001],
    [2016],
    [2012],
    [2017],
    [2011],
    [2014],
    [2008],
    [2012],
    [2015],
    [2017]])
print x
print y
print test


pyplot.scatter(x,y)
pyplot.xlabel('Modelo')
pyplot.ylabel('Precio')
# pyplot.show()

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(test, y)

print reg.predict([[2011]])
print reg.score(test, y )

pyplot.plot(test, reg.predict(test))
pyplot.show()