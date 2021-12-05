import numpy as np

# download the data and transform into .txt file.
filename = 'hw1_train.dat.txt'

# the question asks us to take sign(0) = -1.
def sign(x):
    if x > 0:
        return 1
    return -1

# set data X, the value of x0 depends on the questions.
def set_data(value_x0):
    data = np.genfromtxt(filename)
    width, height = data.shape
    data_set = np.zeros((width, height))
    
    for i in range(width):
        data_set[i][0] = value_x0
        data_set[i, 1:height] = data[i, 0:height-1]
        
    return data_set

# set data label.
def set_label():
    data = np.genfromtxt(filename)
    width, height = data.shape
    label_set = np.zeros((width, 1))
    
    for i in range(width):
        label_set[i][0] = data[i][height - 1]
        
    return label_set

# the process of PLA algorithm.
def PLA(X, y, normalization):
    wt = np.repeat(0, len(X[0]))  # initial wt = 0.
    correct = 0  # number of consecutive correct counts.
    square_length = 0  # square length of wPLA.
    
    flag = True
    
    while flag:
        seed = np.random.randint(low = 0, high = 99, size = 1)  # random num [0, 99].
        xn = X[seed].reshape(-1)
    
        if (normalization == True):  # if the question asks us to normalize xn.
            xn_length = (np.sum(np.square(xn)))**0.5
            xn = xn / xn_length
        
        yn = y[seed].reshape(-1)
        
        # PLA update and correct process.
        if sign(wt.dot(xn)) != yn:
            correct = 0
            wt = wt + yn * xn
            
        elif sign(wt.dot(xn)) == yn:
            correct += 1
            
        #  Stop updating and return wt as wPLA if wt is correct consecutively after
        #  checking 5N randomly-picked examples. N = 100.
        if correct >= 500:  # termination condition.
            square_length = np.sum(np.square(wt))
            flag = False
            
            return square_length

# main program to executive PLA.
def executive_PLA(value_x0 = 1, scale_para = 1, exp_times = 1000, normalization = False):

    X = set_data(value_x0)
    y = set_label()
    
    X *= scale_para
    store = np.array([PLA(X, y, normalization) for i in range(exp_times)])
    mean_square_length = np.mean(store)  # calculate average squared length of wPLA.
    print(mean_square_length)  # show result.
 
# scale_para = 1, which means we don't do anything.

# 13. [b]
# value_x0 = 1, scale_para = 1, run exp_times = 1000, normalization = False.
executive_PLA()

# 14. [c]
# value_x0 = 1, scale_para = 2, run exp_times = 1000, normalization = False.
executive_PLA(scale_para = 2)

# 15. [e]
# value_x0 = 1, scale_para = 1, run exp_times = 1000, normalization = True.
executive_PLA(normalization = True)

# 16. [a]
# value_x0 = 0, scale_para = 1, run exp_times = 1000, normalization = False.
executive_PLA(value_x0 = 0)
