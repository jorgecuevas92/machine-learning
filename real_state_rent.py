#!/usr/bin/env python

"""This module uses machine learning to predict real state rent based on the squared meters and bathrooms"""
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
    apartments = json.load(open('./apartments.json'))

    # Define the numpy arrays for squared meters, bathrooms and price values.

    sq_meters = np.array([])

    bathrooms = np.array([])

    half_bathrooms = np.array([])

    parking_lots = np.array([])

    ages = np.array([])

    prices = np.array([])

    # Assign values for each array.
    for apartment in apartments:
        sq_meters = np.append(sq_meters, int(apartment['squared_meters']))
        bathrooms = np.append(bathrooms, int(apartment['bathrooms']))
        half_bathrooms = np.append(half_bathrooms, int(apartment['half_bathrooms']))
        parking_lots = np.append(parking_lots, int(apartment['parking_lots']))
        ages = np.append(ages, int(apartment['years']))
        prices = np.append(prices, int(apartment['price']))
    
    # Define the training features array from the squared meters
    # and bathrooms.
    features_train = np.array([sq_meters, bathrooms, half_bathrooms, parking_lots, ages]).T

    print features_train

    # Define the linear regression.
    reg = linear_model.LinearRegression()

    # Train it with the features and target values.
    reg.fit(features_train, prices)

    # prediction = reg.predict([[2011, 23000], [2005, 35000], [2017, 50]])
    # print prediction

    print "Accuracy: " + str(reg.score(features_train, prices))

    # Add 3D plot for the data.
    fig = pyplot.figure()

    ax = fig.add_subplot(111, projection='3d')

    # Add dots for the training data.
    ax.scatter(sq_meters, bathrooms, prices, label='Training Set')

    # Add dots for the prediction data
    # ax.scatter([2011, 2005, 2007], [23000, 35000, 50], prediction, label='Predictions')
    ax.legend()
    ax.set_xlabel('Squared Meters')
    ax.set_ylabel('Bathrooms')
    ax.set_zlabel('Value')

    # Show the plot results.
    pyplot.show()

if __name__ == "__main__":
    main()
