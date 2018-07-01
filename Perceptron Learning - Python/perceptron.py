import math
import csv
import random
import matplotlib.pyplot as plot


with open('D:/workspace/USU Assignment/USU-Assignment/CS6600 - Intelligent Systems/Assignment 3/Assignment 3-Python/data.csv', 'r') as file:
    reader = csv.reader(file)
    data_set = list(reader)


def sigmoid_function(weight0, weight1, weight2, input1, input2):
    h = weight0 + (weight1 * input1) + (weight2 * input2)
    return float(1) / (1 + math.exp(-1 * h))


def calculate_error(target_output, actual_output):
    return math.pow((target_output - actual_output), 2)


def derivate_sigmoid_function(sigmoid_function_output):
    return sigmoid_function_output * (1 - sigmoid_function_output)


def calculateDeltaWeight(sigmoid_function_output, input_value, target_output, learning_rate=0.001):
    derivative = derivate_sigmoid_function(sigmoid_function_output)
    return -1 * learning_rate * (sigmoid_function_output - target_output) * derivative * input_value


averageError = []


def train(data):
    w0 = random.uniform(-0.2, 0.2)
    w1 = random.uniform(-0.2, 0.2)
    w2 = random.uniform(-0.2, 0.2)

    average_error = 100
    iteration = 0
    while iteration < 1000 and average_error > 0.0001:
        total_error = 0

        for row in data:
            x1 = float(row[0])
            x2 = float(row[1])
            target_output = float(row[2])

            sigmoid_function_output = sigmoid_function(w0, w1, w2, x1, x2)

            error = calculate_error(target_output, sigmoid_function_output)
            total_error += error

            w0 += calculateDeltaWeight(sigmoid_function_output, 1, target_output, 0.001)
            w1 += calculateDeltaWeight(sigmoid_function_output, x1, target_output, 0.001)
            w2 += calculateDeltaWeight(sigmoid_function_output, x2, target_output, 0.001)

        iteration += 1
        average_error = float(total_error) / len(data)
        averageError.append(average_error)

    return {'w0': w0, 'w1': w1, 'w2': w2}


def get_accuracy(test_data, weights):
    matched = 0
    for test_case in test_data:
        test_input1 = float(test_case[0])
        test_input2 = float(test_case[1])
        expected_output = float(test_case[2])

        sigmoid_function_output = sigmoid_function(weights['w0'], weights['w1'], weights['w2'], test_input1, test_input2)
        if sigmoid_function_output <= 0.5:
            predicted_output = 0
        else:
            predicted_output = 1

        if expected_output == predicted_output:
            matched += 1
    return float(matched) / len(test_data) * 100


def ten_fold_validation(data_set):
    total_accuracy = 0
    step = 10
    for start in range(1, len(data_set), step):
        test_set = []
        training_set = []

        for index in range(1, len(data_set)):
            if start <= index < (start + step):
                test_set.append(data_set[index])
            else:
                training_set.append(data_set[index])
        weights = train(training_set)
        accuracy = get_accuracy(test_set, weights)
        total_accuracy += accuracy
        print("Accuracy = " + str(accuracy))
    print("Average accuracy: " + str(float(total_accuracy) / step))


ten_fold_validation(data_set)
plot.plot(averageError)
plot.title('General Data')
plot.xlabel('epoch')
plot.ylabel('error')
plot.show()


# AND gate
with open('D:/workspace/USU Assignment/USU-Assignment/CS6600 - Intelligent Systems/Assignment 3/Assignment 3-Python/and_data.csv', 'r') as file1:
    reader = csv.reader(file1)
    train_and = list(reader)

averageError = []

and_weights = train(train_and)
plot.plot(averageError)
plot.title('AND Gate')
plot.xlabel('epoch')
plot.ylabel('error')
plot.show()

#OR Gate
with open('D:/workspace/USU Assignment/USU-Assignment/CS6600 - Intelligent Systems/Assignment 3/Assignment 3-Python/or_data.csv', 'r') as file2:
    reader = csv.reader(file2)
    train_or = list(reader)

averageError = []

or_weights = train(train_or)
plot.plot(averageError)
plot.title('OR Gate')
plot.xlabel('epoch')
plot.ylabel('error')
plot.show()

#NAND Gate
with open('D:/workspace/USU Assignment/USU-Assignment/CS6600 - Intelligent Systems/Assignment 3/Assignment 3-Python/nand_data.csv', 'r') as file3:
    reader = csv.reader(file3)
    train_nand = list(reader)

averageError = []

nand_weights = train(train_nand)
plot.plot(averageError)
plot.title('NAND Gate')
plot.xlabel('epoch')
plot.ylabel('error')
plot.show()

#NOR Gate
with open('D:/workspace/USU Assignment/USU-Assignment/CS6600 - Intelligent Systems/Assignment 3/Assignment 3-Python/nor_data.csv', 'r') as file4:
    reader = csv.reader(file4)
    train_nor = list(reader)

averageError = []

nor_weights = train(train_nor)
plot.plot(averageError)
plot.title('NOR Gate')
plot.xlabel('epoch')
plot.ylabel('error')
plot.show()

#XOR Gate
with open('D:/workspace/USU Assignment/USU-Assignment/CS6600 - Intelligent Systems/Assignment 3/Assignment 3-Python/xor_data.csv', 'r') as file5:
    reader = csv.reader(file5)
    train_xor = list(reader)

averageError = []

xor_weights = train(train_xor)
plot.plot(averageError)
plot.title('XOR Gate')
plot.xlabel('epoch')
plot.ylabel('error')
plot.show()