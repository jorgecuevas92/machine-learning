#!/usr/bin/env python

"""This module uses machine learning to predict motorcycle prices based on the year and mileage in kilometers."""
import json
import numpy as np
import matplotlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model

def main():
    """This function executes the main program."""

    # Prevent scientific notation on numpy arrays for this example.
    np.set_printoptions(suppress=True)

    # Import existing motorcycle objects from the file.
    motorcycles = json.load(open('./motorcycles.json'))

    # Define the numpy arrays for years, kilometers and prices.
    year_values = np.array([])

    km_values = np.array([])

    price_values = np.array([])

    # Assign the values for each motorcycle.
    for moto in motorcycles:
        year_values = np.append(year_values, int(moto['year']))
        km_values = np.append(km_values, int(moto['km']))
        price_values = np.append(price_values, int(moto['price']))

    # Define the training features array  from the years and km
    # arrays and transpose it.
    features_train = np.array([year_values, km_values]).T

    print year_values
    print price_values
    print features_train

    # Plot the dots for price as a function of years.
    pyplot.scatter(year_values, price_values)
    pyplot.xlabel('Modelo')
    pyplot.ylabel('Precio')

    # Define the linear regression.
    reg = linear_model.LinearRegression()

    # Train it with the features and target values.
    reg.fit(features_train, price_values)

    # Predict the price based on year and kilometers: [[years, km]].
    print reg.predict([[2011, 23000]])

    # Print the accuracy of the prediction model based on the training set.
    print reg.score(features_train, price_values)

    # Plot the prediction model.
    # pyplot.plot(year_values, reg.predict(features_train))

    print reg.predict(features_train)

    # Add 3D plot for the data.
    fig = pyplot.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(year_values, km_values, price_values, label='parametric curve')
    ax.legend()
    ax.set_xlabel('Year')
    ax.set_ylabel('Kilometers')
    ax.set_zlabel('Value')
    
    # Show the plot results.
    pyplot.show()


if __name__ == "__main__":
    main()
