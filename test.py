import logisticRegression as lr

# Load training set and test set
trainSet,labelSet = lr.load_train_data('lr_data.txt')
testSet,testLabelSet = lr.load_test_data('lr_data.txt')


# Apply Logistic Regression to get the optimal weights
weight = lr.logistic_regression(trainSet, labelSet, 1000, 0.001)

# Run the test data to compute the prediction and precision
accuracy = lr.fit(testSet, testLabelSet, weight)

print(accuracy)

