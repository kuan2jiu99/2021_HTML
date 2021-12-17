import numpy as np
import math
import random

# import training and testing data.
train_filename = 'hw3_train.dat.txt'
test_filename = 'hw3_test.dat.txt'


def set_data(filename):
    data = np.genfromtxt(filename)
    width, height = data.shape
    
    data_set = np.zeros((width, height - 1))
    
    for i in range(width):
        data_set[i, 0:height - 1] = data[i, 0:height - 1]
    
    return data_set
    
def set_label(filename):
    data = np.genfromtxt(filename)
    width, height = data.shape
    
    label_set = np.zeros((width, 1))
    
    for i in range(width):
        label_set[i][0] = data[i][height - 1]
    
    return label_set

def polynomial_transform(data, Q_value):
    width, height = data.shape
    array_size = 1 + Q_value * 10
    trans_data = np.ones((width, array_size))
    
    for i in range(width):
        for q_index in range(1, Q_value + 1):
            for j in range((q_index - 1) * 10 + 1, (q_index - 1) * 10 + 11):
                if (q_index != 1):
                    trans_data[i][j] = (data[i][j - (q_index - 1) * 10 - 1])**(q_index)
                else:
                    trans_data[i][j] = (data[i][j - 1])
                    
    return trans_data

def zero_one_error(y, y_predict):
    width, height = y_predict.shape  # (1x1000)
    error = 0
    for i in range(height):
        if (y[0][i] != y_predict[0][i]):
            error += 1
    
    return error / height
    

training_data = set_data(train_filename)
training_label = set_label(train_filename)
testing_data = set_data(test_filename)
testing_label = set_label(test_filename)


# (12). Ans = 0.326, [b].
train_12 = polynomial_transform(training_data, 2)
test_12 = polynomial_transform(testing_data, 2)

# train: linear regression.
wt = np.linalg.pinv(train_12.T @ train_12) @(train_12.T @ training_label) # (21x1)

# calculate 1/0 error.
y_train_predict = np.array([[1 if y >= 0 else -1 for y in (train_12 @ wt)]])  # (1x1000)

y_test_predict = np.array([[1 if y >= 0 else -1 for y in (test_12 @ wt)]])

Ein_01 = zero_one_error(training_label.T, y_train_predict)

Eout_01 = zero_one_error(testing_label.T, y_test_predict)

print(abs(Ein_01 - Eout_01))

 

# (13). Ans = 0.4576, [d]
train_13 = polynomial_transform(training_data, 8)
test_13 = polynomial_transform(testing_data, 8)

# train: linear regression.
wt = np.linalg.pinv(train_13.T @ train_13) @(train_13.T @ training_label) # (21x1)

# calculate 1/0 error.
y_train_predict = np.array([[1 if y >= 0 else -1 for y in (train_13 @ wt)]])
y_test_predict = np.array([[1 if y >= 0 else -1 for y in (test_13 @ wt)]])

Ein_01 = zero_one_error(training_label.T, y_train_predict)
Eout_01 = zero_one_error(testing_label.T, y_test_predict)

print(abs(Ein_01 - Eout_01))


def polynomial_transform_2(data):
    width, height = data.shape
    array_size = 66  # 66.
    trans_data = np.ones((width, array_size))
    Q_value = 2
    
    for i in range(width):
        for j in range(1, 10 + 1):
            trans_data[i][j] = data[i][j - 1]
            
    
    for i in range(width):
        k = 0
        for j in range(11, 21): # 10
            trans_data[i][j] = data[i][0] * data[i][0 + k]
            k += 1
    
    for i in range(width):
        k = 0
        for j in range(21, 30): # 9
            trans_data[i][j] = data[i][1] * data[i][1 + k]
            k += 1
    
    for i in range(width):
        k = 0
        for j in range(30, 38): # 8
            trans_data[i][j] = data[i][2] * data[i][2 + k]
            k += 1
    
    for i in range(width):
        k = 0
        for j in range(38, 45): # 7
            trans_data[i][j] = data[i][3] * data[i][3 + k]
            k += 1
    
    for i in range(width):
        k = 0
        for j in range(45, 51): # 6
            trans_data[i][j] = data[i][4] * data[i][4 + k]
            k += 1
        
    for i in range(width):
        k = 0
        for j in range(51, 56): # 5
            trans_data[i][j] = data[i][5] * data[i][5 + k]
            k += 1
       
    for i in range(width):
        k = 0
        for j in range(56, 60): # 4
            trans_data[i][j] = data[i][6] * data[i][6 + k]
            k += 1
        
    for i in range(width):
        k = 0
        for j in range(60, 63): # 3
            trans_data[i][j] = data[i][7] * data[i][7 + k]
            k += 1
       
    for i in range(width):
        k = 0
        for j in range(63, 65): # 2
            trans_data[i][j] = data[i][8] * data[i][8 + k]
            k += 1
       
    for i in range(width):
        k = 0
        for j in range(65, 66): # 1
            trans_data[i][j] = data[i][9] * data[i][9 + k]
            k += 1
                    
    return trans_data

# (14). Ans = 0.338, [a].
train_14 = polynomial_transform_2(training_data)
test_14 = polynomial_transform_2(testing_data)

# train: linear regression.
wt = np.linalg.pinv(train_14.T @ train_14) @(train_14.T @ training_label)

# calculate 1/0 error.
y_train_predict = np.array([[1 if y >= 0 else -1 for y in (train_14 @ wt)]])  # (1x1000)

y_test_predict = np.array([[1 if y >= 0 else -1 for y in (test_14 @ wt)]])

Ein_01 = zero_one_error(training_label.T, y_train_predict)

Eout_01 = zero_one_error(testing_label.T, y_test_predict)

print(abs(Ein_01 - Eout_01))

def lower_dim_transformation(data, low_dim):
    width, height = data.shape
    trans_data = np.ones((width, low_dim + 1))
    for i in range(width):
        trans_data[i, 1:low_dim + 1] = data[i, 0:low_dim]
    
    return trans_data

#(15). Ans = 3, [c].
record = 0
min_abs_E = 0
for i in range(1, 1 + 1):
    train_15 = lower_dim_transformation(training_data, i)
    test_15 = lower_dim_transformation(testing_data, i)

    # train: linear regression.
    wt = np.linalg.pinv(train_15.T @ train_15) @(train_15.T @ training_label)

    # calculate 1/0 error.
    y_train_predict = np.array([[1 if y >= 0 else -1 for y in (train_15 @ wt)]])

    y_test_predict = np.array([[1 if y >= 0 else -1 for y in (test_15 @ wt)]])

    Ein_01 = zero_one_error(training_label.T, y_train_predict)

    Eout_01 = zero_one_error(testing_label.T, y_test_predict)
    
    min_abs_E = abs(Ein_01 - Eout_01)
    print(min_abs_E)
    record = 1

for i in range(2, 10 + 1):
    train_15 = lower_dim_transformation(training_data, i)
    test_15 = lower_dim_transformation(testing_data, i)

    # train: linear regression.
    wt = np.linalg.pinv(train_15.T @ train_15) @(train_15.T @ training_label)

    # calculate 1/0 error.
    y_train_predict = np.array([[1 if y >= 0 else -1 for y in (train_15 @ wt)]])

    y_test_predict = np.array([[1 if y >= 0 else -1 for y in (test_15 @ wt)]])

    Ein_01 = zero_one_error(training_label.T, y_train_predict)

    Eout_01 = zero_one_error(testing_label.T, y_test_predict)
    print(abs(Ein_01 - Eout_01))
    
    if (abs(Ein_01 - Eout_01) <= min_abs_E):
        record = i
        min_abs_E = abs(Ein_01 - Eout_01)
        
print(record)

    



def random_choose_transformation(data, random_array):
    width, height = data.shape
    trans_data = np.ones((width, 6))
    
    for i in range(width):
        trans_data[i][1] = data[i][random_array[0] - 1]
        trans_data[i][2] = data[i][random_array[1] - 1]
        trans_data[i][3] = data[i][random_array[2] - 1]
        trans_data[i][4] = data[i][random_array[3] - 1]
        trans_data[i][5] = data[i][random_array[4] - 1]
        
    return trans_data


# (16). Ans = 0.206, [d].
E_avg = 0
for iterations in range(200):
    random_choose = np.array(random.sample(range(1, 11), 5))
    
    train_16 = random_choose_transformation(training_data, random_choose)
    test_16 = random_choose_transformation(testing_data, random_choose)
    
    # train: linear regression.
    wt = np.linalg.pinv(train_16.T @ train_16) @(train_16.T @ training_label)

    # calculate 1/0 error.
    y_train_predict = np.array([[1 if y >= 0 else -1 for y in (train_16 @ wt)]])

    y_test_predict = np.array([[1 if y >= 0 else -1 for y in (test_16 @ wt)]])

    Ein_01 = zero_one_error(training_label.T, y_train_predict)

    Eout_01 = zero_one_error(testing_label.T, y_test_predict)
    
    E_avg += abs(Ein_01 - Eout_01)
    
print(E_avg / 200)


