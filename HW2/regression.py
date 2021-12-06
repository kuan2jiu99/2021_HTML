import numpy as np
import random
import math

# (13), ans = 0.28, [d].
# training data

Ein = 0
for runtimes in range(100):
    np.random.seed()
    flip_coin = np.random.randint(2, size = 200)  # 1 for head, 0 for tail.
    y_label = np.ones(200)
    for i in range(200):
        if flip_coin[i] == 0:
            y_label[i] = -1
    
    vector_x_train = np.ones((200, 3))

    for i in range(200):
        if (y_label[i] == 1):
            mean = [2, 3]
            cov = [[0.6, 0], [0, 0.6]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_train[i][1] = x1
            vector_x_train[i][2] = x2
        else:
            mean = [0, 4]
            cov = [[0.4, 0], [0, 0.4]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_train[i][1] = x1
            vector_x_train[i][2] = x2

    wt = np.linalg.inv(vector_x_train.T@vector_x_train)@(vector_x_train.T@y_label.T)

    y_predict = wt @ (vector_x_train.T)  # 1x200
    Ein += np.mean((y_label - y_predict)**2)

print(Ein / 100)


# (14), ans = 0.013, [d].

Ein = 0
Eout = 0
E_abs = 0
for runtimes in range(100):
    # training data.
    
    flip_coin = np.random.randint(2, size = 200)  # 1 for head, 0 for tail.
    y_label = np.ones(200)
    
    for i in range(200):
        if flip_coin[i] == 0:
            y_label[i] = -1
    
    vector_x_train = np.ones((200, 3))

    for i in range(200):
        if (y_label[i] == 1):
            mean = [2, 3]
            cov = [[0.6, 0], [0, 0.6]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_train[i][1] = x1
            vector_x_train[i][2] = x2
        else:
            mean = [0, 4]
            cov = [[0.4, 0], [0, 0.4]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_train[i][1] = x1
            vector_x_train[i][2] = x2

    wt = np.linalg.inv(vector_x_train.T@vector_x_train)@(vector_x_train.T@y_label.T)

    y_predict = wt @ (vector_x_train.T) # 1x200
    y_predict = y_predict.reshape(1, 200)
    y_train_predict = [1 if y >= 0 else -1 for y in y_predict[0]]
    
    for j in range(200):
        if y_train_predict[j] != y_label[j]:
            Ein += 1
    Ein = Ein / 200
    
    # testing data.
    
    flip_coin_test = np.random.randint(2, size = 5000)
    y_label_test = np.ones(5000)
    
    for i in range(5000):
        if flip_coin_test[i] == 0:
            y_label_test[i] = -1
            
    
    vector_x_test = np.ones((5000, 3))

    for i in range(5000):
        if (y_label_test[i] == 1):
            mean = [2, 3]
            cov = [[0.6, 0], [0, 0.6]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_test[i][1] = x1
            vector_x_test[i][2] = x2
        else:
            mean = [0, 4]
            cov = [[0.4, 0], [0, 0.4]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_test[i][1] = x1
            vector_x_test[i][2] = x2
            
    y_predict_test = wt @ (vector_x_test.T)
    y_predict_test = y_predict_test.reshape(1, 5000)
    y_train_predict = [1 if y >= 0 else -1 for y in y_predict_test[0]]
    
    for j in range(5000):
        if y_train_predict[j] != y_label_test[j]:
            Eout += 1
    Eout = Eout / 5000
    
    E_abs += abs(Ein - Eout)
    Ein = 0
    Eout = 0

print(E_abs / 100)



# (15), ans = (0.058, 0.058), [b]

# Part of Logistic Regression. Ans = 0.057
def logistic(s):
    return (1 + math.exp(-s))**(-1)
    
def gradients(x, y, N, wt):
    gradient = np.zeros((1, 3))
    for i in range(N):
        x_train = x[i]  # size : (1x3)
        y_train = y[i]
        gradient += logistic(-y_train * (wt@x[i].T)) * (-y_train * x_train)
    
    gradient /= N
    
    return gradient

wt = np.repeat(0, 3)  # initial w0
eta = 0.1
Eout = 0
average_Eout = 0
for runtimes in range(100):
    Eout = 0
    for iterations in range(500):
        # training data.
        
        flip_coin = np.random.randint(2, size = 200)  # 1 for head, 0 for tail.
        y_label = np.ones(200)
        
        for i in range(200):
            if flip_coin[i] == 0:
                y_label[i] = -1
        
        vector_x_train = np.ones((200, 3))

        for i in range(200):
            if (y_label[i] == 1):
                mean = [2, 3]
                cov = [[0.6, 0], [0, 0.6]]
                x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
                vector_x_train[i][1] = x1
                vector_x_train[i][2] = x2
            else:
                mean = [0, 4]
                cov = [[0.4, 0], [0, 0.4]]
                x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
                vector_x_train[i][1] = x1
                vector_x_train[i][2] = x2

        
        wt = wt - eta * gradients(vector_x_train, y_label, 200, wt)

    # testing data.
    
    flip_coin_test = np.random.randint(2, size = 5000)
    y_label_test = np.ones(5000)
    
    for i in range(5000):
        if flip_coin_test[i] == 0:
            y_label_test[i] = -1
            
    
    vector_x_test = np.ones((5000, 3))

    for i in range(5000):
        if (y_label_test[i] == 1):
            mean = [2, 3]
            cov = [[0.6, 0], [0, 0.6]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_test[i][1] = x1
            vector_x_test[i][2] = x2
        else:
            mean = [0, 4]
            cov = [[0.4, 0], [0, 0.4]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_test[i][1] = x1
            vector_x_test[i][2] = x2
            
    y_predict_test = wt @ (vector_x_test.T)
    y_predict_test = y_predict_test.reshape(1, 5000)
    y_train_predict = [1 if y >= 0.5 else -1 for y in y_predict_test[0]]
    
    for j in range(5000):
        if y_train_predict[j] != y_label_test[j]:
            Eout += 1
    Eout = Eout / 5000
    
    average_Eout += Eout
    
    
print(average_Eout / 100)


# (15)
# Part of Linear Regression, ans = 0.058
Ein = 0
Eout = 0
average_Eout = 0
for runtimes in range(100):
    Eout = 0
    # training data.
    
    flip_coin = np.random.randint(2, size = 200)  # 1 for head, 0 for tail.
    y_label = np.ones(200)
    
    for i in range(200):
        if flip_coin[i] == 0:
            y_label[i] = -1
    
    vector_x_train = np.ones((200, 3))

    for i in range(200):
        if (y_label[i] == 1):
            mean = [2, 3]
            cov = [[0.6, 0], [0, 0.6]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_train[i][1] = x1
            vector_x_train[i][2] = x2
        else:
            mean = [0, 4]
            cov = [[0.4, 0], [0, 0.4]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_train[i][1] = x1
            vector_x_train[i][2] = x2

    wt = np.linalg.inv(vector_x_train.T@vector_x_train)@(vector_x_train.T@y_label.T)
    
    # testing data.
    
    flip_coin_test = np.random.randint(2, size = 5000)
    y_label_test = np.ones(5000)
    
    for i in range(5000):
        if flip_coin_test[i] == 0:
            y_label_test[i] = -1
            
    
    vector_x_test = np.ones((5000, 3))

    for i in range(5000):
        if (y_label_test[i] == 1):
            mean = [2, 3]
            cov = [[0.6, 0], [0, 0.6]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_test[i][1] = x1
            vector_x_test[i][2] = x2
        else:
            mean = [0, 4]
            cov = [[0.4, 0], [0, 0.4]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_test[i][1] = x1
            vector_x_test[i][2] = x2
            
    y_predict_test = wt @ (vector_x_test.T)
    y_predict_test = y_predict_test.reshape(1, 5000)
    y_train_predict = [1 if y >= 0 else -1 for y in y_predict_test[0]]
    
    for j in range(5000):
        if y_train_predict[j] != y_label_test[j]:
            Eout += 1
    Eout = Eout / 5000
    average_Eout += Eout
    

print(average_Eout / 100)


# (16). Ans = (0.090, 0.058), [c]
# Part of Logistic Regression. Ans = 0.058
# Add 20 outlier examples.

def logistic(s):
    return (1 + math.exp(-s))**(-1)
    
def gradients(x, y, N, wt):
    gradient = np.zeros((1, 3))
    for i in range(N):
        x_train = x[i]  # size : (1x3)
        y_train = y[i]
        gradient += logistic(-y_train * (wt@x[i].T)) * (-y_train * x_train)
    
    gradient /= N
    
    return gradient

wt = np.repeat(0, 3)  # initial w0
eta = 0.1
Eout = 0
average_Eout = 0
for runtimes in range(100):
    Eout = 0
    for iterations in range(500):
        # training data.
        
        flip_coin = np.random.randint(2, size = 200)  # 1 for head, 0 for tail.
        y_label = np.ones(200 + 20)
        
        for i in range(200):
            if flip_coin[i] == 0:
                y_label[i] = -1
        
        vector_x_train = np.ones((200 + 20, 3))

        for i in range(200):
            if (y_label[i] == 1):
                mean = [2, 3]
                cov = [[0.6, 0], [0, 0.6]]
                x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
                vector_x_train[i][1] = x1
                vector_x_train[i][2] = x2
            else:
                mean = [0, 4]
                cov = [[0.4, 0], [0, 0.4]]
                x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
                vector_x_train[i][1] = x1
                vector_x_train[i][2] = x2
        
        for i in range(20):
                mean = [6, 0]
                cov = [[0.3, 0], [0, 0.1]]
                x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
                vector_x_train[i + 200][1] = x1
                vector_x_train[i + 200][2] = x2
        
        wt = wt - eta * gradients(vector_x_train, y_label, 200 + 20, wt)

    # testing data.
    
    flip_coin_test = np.random.randint(2, size = 5000)
    y_label_test = np.ones(5000)
    
    for i in range(5000):
        if flip_coin_test[i] == 0:
            y_label_test[i] = -1
            
    
    vector_x_test = np.ones((5000, 3))

    for i in range(5000):
        if (y_label_test[i] == 1):
            mean = [2, 3]
            cov = [[0.6, 0], [0, 0.6]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_test[i][1] = x1
            vector_x_test[i][2] = x2
        else:
            mean = [0, 4]
            cov = [[0.4, 0], [0, 0.4]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_test[i][1] = x1
            vector_x_test[i][2] = x2
            
    y_predict_test = wt @ (vector_x_test.T)
    y_predict_test = y_predict_test.reshape(1, 5000)
    y_train_predict = [1 if y >= 0.5 else -1 for y in y_predict_test[0]]
    
    for j in range(5000):
        if y_train_predict[j] != y_label_test[j]:
            Eout += 1
    Eout = Eout / 5000
    
    average_Eout += Eout
    
    
print(average_Eout / 100)

# (16)
# Part of Linear Regression, ans = 0.090

Ein = 0
Eout = 0
average_Eout = 0
for runtimes in range(100):
    Eout = 0
    # training data.
    
    flip_coin = np.random.randint(2, size = 200)  # 1 for head, 0 for tail.
    y_label = np.ones(200 + 20)
    
    for i in range(200):
        if flip_coin[i] == 0:
            y_label[i] = -1
    
    vector_x_train = np.ones((200 + 20, 3))

    for i in range(200):
        if (y_label[i] == 1):
            mean = [2, 3]
            cov = [[0.6, 0], [0, 0.6]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_train[i][1] = x1
            vector_x_train[i][2] = x2
        else:
            mean = [0, 4]
            cov = [[0.4, 0], [0, 0.4]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_train[i][1] = x1
            vector_x_train[i][2] = x2
            
    for i in range(20):
            mean = [6, 0]
            cov = [[0.3, 0], [0, 0.1]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_train[i + 200][1] = x1
            vector_x_train[i + 200][2] = x2

    wt = np.linalg.inv(vector_x_train.T@vector_x_train)@(vector_x_train.T@y_label.T)
    
    # testing data.
    
    flip_coin_test = np.random.randint(2, size = 5000)
    y_label_test = np.ones(5000)
    
    for i in range(5000):
        if flip_coin_test[i] == 0:
            y_label_test[i] = -1
            
    
    vector_x_test = np.ones((5000, 3))

    for i in range(5000):
        if (y_label_test[i] == 1):
            mean = [2, 3]
            cov = [[0.6, 0], [0, 0.6]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_test[i][1] = x1
            vector_x_test[i][2] = x2
        else:
            mean = [0, 4]
            cov = [[0.4, 0], [0, 0.4]]
            x1, x2 = np.random.multivariate_normal(mean , cov, 1).T
            vector_x_test[i][1] = x1
            vector_x_test[i][2] = x2
            
    y_predict_test = wt @ (vector_x_test.T)
    y_predict_test = y_predict_test.reshape(1, 5000)
    y_train_predict = [1 if y >= 0 else -1 for y in y_predict_test[0]]
    
    for j in range(5000):
        if y_train_predict[j] != y_label_test[j]:
            Eout += 1
    Eout = Eout / 5000
    average_Eout += Eout
    
print(average_Eout / 100)

