# Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.set_printoptions(suppress=True)
rng = np.random.default_rng(seed = 42) # Allows for consistent reproducibility


# Constant for the acceleration due to gravity
G = 9.81
MIN_VELOCITY = 10
MAX_VELOCITY = 50
NUM_TEST_SAMPLES = 250
NUM_TRAINING_SAMPLES = 1000
MIN_ANGLE = 0
MAX_ANGLE = 90

def main():
    initial_velocities = rng.integers(MIN_VELOCITY, MAX_VELOCITY, NUM_TRAINING_SAMPLES) # Initial velocities training data
    degrees = rng.integers(MIN_ANGLE, MAX_ANGLE, NUM_TRAINING_SAMPLES) # Launch angles training data
    rad_angles = np.deg2rad(degrees) # Convert launch angles into radians
    labels = (initial_velocities ** 2 * np.sin(rad_angles) ** 2) / (2 * G), (initial_velocities ** 2 * np.sin(2 * rad_angles)) / G # Test data for the correct outputs, uses equations to find max height and range


    # Stack the velocities and angles into a 2D array to be accepted by the model, do the same for the max heights and ranges
    initial_conditions = np.column_stack((initial_velocities, rad_angles))
    properties = np.column_stack(labels) # Properties of the projectile motion - max height and range


    # First using standard Linear Regression
    model = LinearRegression()
    model.fit(initial_conditions, properties)


    # All the test data
    test_initial_velocities = rng.integers(MIN_VELOCITY, MAX_VELOCITY, NUM_TEST_SAMPLES)
    test_degrees = rng.integers(MIN_ANGLE, MAX_ANGLE, NUM_TEST_SAMPLES)
    test_rad_angles = np.deg2rad(test_degrees)
    test_properties = np.column_stack(((test_initial_velocities ** 2 * np.sin(test_rad_angles) ** 2) / (2 * G), (test_initial_velocities ** 2 * np.sin(2 * test_rad_angles)) / G))
    test_initial_conditions = np.column_stack((test_initial_velocities, test_rad_angles))

    print(f"Score for the model using Linear Regression (3 DPs): {round(model.score(test_initial_conditions, test_properties), 3)}")



    # Using polynomial regression
    poly = PolynomialFeatures(degree = 2, include_bias = True)

    initial_conditions_poly = poly.fit_transform(initial_conditions) # fit() determines the optimal coefficients for the polynomial equation, while transform() applies the transformation from fit()
    test_conditions_poly = poly.transform(test_initial_conditions) # Do not fit this data, you only fit the training data to find the optimal coefficients, here you already have the coefficients
    # The output y values do not need to be changed at all because we need those exact values in that format to test the model

    # From here, just train the model as normal as if using linear regression
    model.fit(initial_conditions_poly, properties)
    predictions = model.predict(test_conditions_poly)
    print(f"Score for the model using Polynomial Regression (3 DPs): {round(model.score(test_conditions_poly, test_properties), 3)}")


    # Showing how the predicted height compares with the actual max height
    plt.scatter(test_properties[:,0], predictions[:,0], label="Max Height", marker = "x")
    plt.xlabel("Actual Max Height")
    plt.ylabel("Predicted Max Height")
    plt.title("Polynomial Regression Predictions")

    plt.legend()
    plt.show()

    # Showing how the predicted range compares with the actual range
    plt.scatter(test_properties[:,1], predictions[:,1], label="Range", color = "orange")
    plt.xlabel("Actual Range")
    plt.ylabel("Predicted Range")
    plt.title("Polynomial Regression Predictions")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()