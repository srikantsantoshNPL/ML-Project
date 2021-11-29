from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import seaborn as sns
DIR = 'C:/Users/ss38/Rafa AI stuff/210104'

def populate_samples(x, multiplier=4*4, draw=False):
    # increasing number of samples by taking parts of image
    multiplier = int(multiplier ** 0.5)
    examples = []
    if draw:
        fig, ax1 = plt.subplots(multiplier, multiplier)
    for i in range(multiplier):
        for j in range(multiplier):
            newx = x[i: len(x) + i - multiplier + 1, j: len(x) + j - multiplier + 1]
            examples.append(newx)
            if draw:
                ax1[i, j].imshow(examples[-1])
    if draw: #Not massively sure how this bit works
        plt.show()
    return examples


def structure_all_files(directory, populate=True):
    #group all the files in the dictionary that contains labels (names) and the images
    folders = listdir(directory)
    newfolders = []
    for i in range(len(folders)):  # find folders
        if '.' not in folders[i]:  # eliminate files thart do not have data, what do you mean by the dot
            newfolders.append(folders[i])
    print(newfolders)
    all_ampls = {}
    all_phases = {}
    for folder in newfolders:  # taking files in all (sample) folders
        files = listdir(directory + '\\' + folder)
        print(directory + '\\' + folder)
        grouped_ampls = {}
        grouped_phases = {}
        for f in files:  # for each file we check if it is phase or amplitude
            #place = f.find('for') #When is this needed
            s_freq = f[-9: -4] #What is significant about this range

            if 'ampl' in f:
                if s_freq not in grouped_ampls.keys():
                    grouped_ampls[s_freq] = []   # list initialisation
                x = [np.genfromtxt(directory + '\\' + folder + '\\' + f)]
                # single element list of single data set
                if populate:
                    x = populate_samples(x[0])
                    print(x)
                    #What does the first columns tell you
                for xi in x:  # put all the data from list into category
                    grouped_ampls[s_freq].append(xi)
                    

            if 'phase' in f:  # similar to amplitudes
                if s_freq not in grouped_phases.keys():
                    grouped_phases[s_freq] = []
                x = [np.genfromtxt(directory + '\\' + folder + '\\' + f)]
                if populate:
                    x = populate_samples(x[0])
                for xi in x:
                    grouped_phases[s_freq].append(xi)

        all_ampls[folder] = grouped_ampls
        all_phases[folder] = grouped_phases

    return all_ampls, all_phases


def split_data_vector(samples, frequencies, all_amplitudes, all_phases, r_train, r_val):
    #produces train validation and test data
    for k in range(len(samples)):

        # index types of samples
        number_of_samples = len(all_amplitudes[samples[k]][frequencies[0]])
        for j in range(number_of_samples):  # index sample number
            to_stack_amplitude = []
            to_stack_phase = []
            for i in range(len(frequencies)):  # index frequencies
                to_stack_amplitude.append(all_amplitudes[samples[k]][frequencies[i]][j])
                to_stack_phase.append(all_phases[samples[k]][frequencies[i]][j])
            RGB_amplitude = np.stack(deepcopy(to_stack_amplitude), 2)
            RGB_phase = np.stack(deepcopy(to_stack_phase), 2)
            RGB_amplitude /= np.std(RGB_amplitude)                  # NORMALIZATION
            RGB_amplitude -= np.average(RGB_amplitude)
            RGB_phase /= 360
            if j < r_train * number_of_samples:
                data_train.append([RGB_amplitude, RGB_phase, vectors[k]])
            elif j < (r_train + r_val) * number_of_samples:
                data_val.append([RGB_amplitude, RGB_phase, vectors[k]])
            else:
                data_test.append([RGB_amplitude, RGB_phase, vectors[k]])

        print(samples[k] + ': ' + str(number_of_samples) + ' samples')
    return (data_train, data_val, data_test)

def split_data(samples, frequencies, all_amplitudes, all_phases, r_train, r_val):
    for k in range(len(samples)):  # index types of samples
        number_of_samples = len(all_amplitudes[samples[k]][frequencies[0]])
        for j in range(number_of_samples):  # index sample number
            to_stack_amplitude = []
            to_stack_phase = []
            for i in range(len(frequencies)):  # index frequencies
                to_stack_amplitude.append(all_amplitudes[samples[k]][frequencies[i]][j])
                to_stack_phase.append(all_phases[samples[k]][frequencies[i]][j])
            RGB_amplitude = np.stack(deepcopy(to_stack_amplitude), 2)
            RGB_phase = np.stack(deepcopy(to_stack_phase), 2)
            RGB_amplitude /= np.std(RGB_amplitude)                  # NORMALIZATION
            RGB_amplitude -= np.average(RGB_amplitude)
            RGB_phase /= 360
            if j < r_train * number_of_samples:
                data_train.append([RGB_amplitude, RGB_phase, k])
            elif j < (r_train + r_val) * number_of_samples:
                data_val.append([RGB_amplitude, RGB_phase, k])
            else:
                data_test.append([RGB_amplitude, RGB_phase, k])

        print(samples[k] + ': ' + str(number_of_samples) + ' samples')
    return (data_train, data_val, data_test)


#split
def split_xy_vector(data):
    #splits data to x and y vector for tensorflow
    x = []
    y = []
    for i in range(len(data)):
            x.append(np.concatenate((data[i][0], data[i][1]), 2))
            #x.append(data[i][0])
            y.append(data[i][2])
    x = np.asarray(x)
    y = np.asarray(y)
    print(y)
    if len(y) > 0:
        y = np.asarray(y)
    return x, y

def split_xy(data):
    x = []
    y = []
    for i in range(len(data)):
            x.append(np.concatenate((data[i][0], data[i][1]), 2))
            #x.append(data[i][0])
            y.append(data[i][2])
    x = np.asarray(x)
    y = np.asarray(y)
    if len(y) > 0:
        y = to_categorical(y)
    return x, y


####################################################################

all_amplitudes, all_phases = structure_all_files(DIR)

# Here we specify sample types  and their positions on 'triangle plot' (vectors)
samples  = ['1ferrite', '1copper', 'no_plates', '1co_on_ferrite', '1ferrite_on_co', '1brass']
#add '1steel' normally
vectors = [[1.,0.,0.], [0.,1.,0.], [0., 0., 1.], [.5, .5, 0], [0.9, 0.1, 0], [0,0.5,0.5], [0.2, 0.8, 0]]

# Here we specify which frequencies we want to use
#frequencies = ['3.50', '7.00', '29.0']
frequencies = ['25.25']


####################################### Organising data

data_train = []
data_val = []
data_test = []

# Gathering data

r_train = 0.7
r_val = 0.2
r_test = 0.1

data_train, data_val, data_test = split_data_vector(samples, frequencies, all_amplitudes, all_phases,  r_train, r_val)

random.shuffle(data_train)  
random.shuffle(data_val)

x_train, y_train = split_xy_vector(data_train)
x_val, y_val = split_xy_vector(data_val)
x_test, y_test = split_xy_vector(data_test)


#### Model definition

model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=x_train[0].shape))
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))


model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics='acc')
history = model.fit(x_train, y_train, batch_size=12, epochs=20, validation_data=(x_val, y_val))

# TEST

Y = y_test
Y_AI = model.predict(x_test)
test_loss, test_acc = model.evaluate(x_test, Y)
print('LOSS = '+ str(test_loss) + '      ACC = ' + str(test_acc))
fig, ax1 = plt.subplots(3)
for i in range(3):
    ax1[i].plot(range(len(Y)), Y[:, i], 'ok', range(len(Y)), Y_AI[:, i], 'xr')
plt.show()
Y_AI_classes = np.argmax(Y_AI,axis = 1)
Y_true = np.argmax(Y,axis=1)


all_amplitudes, all_phases = structure_all_files(DIR)
samples = ['1brass',

 '1copper',
 '1co_on_ferrite',
 '1ferrite',
 '1ferrite_on_co',
 
 'no_plates']

colors = ['r', 'b', 'g', 'y', 'm', 'c', 'orange', 'pink', 'grey']
symbols = ['s', 'o', '*', 'D', '^', 'x', '>', '<', 'v']
names = ['brass',  'copper', 'copper on ferrite', 'ferrite', 'ferrite on copper',  'no samples']

####################################### Organising data and predicting


data_train = []
data_val = []
data_test = []

r_train = 0
r_val = 0
r_test = 1

#Applying model to unknown data

data_train, data_val, data_test = split_data(samples, frequencies, all_amplitudes, all_phases,  r_train, r_val)

x_test, y_test = split_xy(data_test)

y_test = np.argmax(y_test, axis=1)

y_AI = model.predict(x_test)

# gathering the data for final plot
jump = False
sigmas = []
mus = []
errsigma = []
errmu = []
sigma = []
mu = []
Xs = []
Ys = []
for i in range(len(y_test)-1):
    jump = y_test[i] != y_test[i+1] # jump if there is new type of sample
    last = i == len(y_test) - 2
    if jump or last:
        sigma.append(y_AI[i][1])
        mu.append(y_AI[i][0])
        Xs.append(mu)
        Ys.append(sigma)
        sigmas.append(np.average(np.asarray(sigma)))
        mus.append(np.average(np.asarray(mu)))
        errsigma.append(np.std(np.asarray(sigma)))
        errmu.append(np.std(np.asarray(mu)))

        sigma = []
        mu = []

    else:
        sigma.append(y_AI[i][1])
        mu.append(y_AI[i][0])


plt.figure()
plt.plot([1, 0, 0, 1], [0, 1, 0, 0], '--k')
for ii in range(len(mus)):
    plt.errorbar(mus[ii], sigmas[ii], errsigma[ii], errmu[ii],
                 ecolor=colors[ii], marker=symbols[ii], mfc=colors[ii], mec='k', label=names[ii], ms=12, fmt='o')

ox = np.arange(0, 1, 0.001)
oy = 1 - ox
plt.legend(loc=1, prop={'size': 9})
plt.ylim((-.1, 1.1))
plt.xlim((-.1, 1.1))
plt.show()
















# plt.figure()
# for ii in range(len(mus)):
#     plt.errorbar(mus[ii], sigmas[ii], errsigma[ii], errmu[ii],
#                  ecolor=symbols[ii][1:], marker=symbols[ii][0], mfc=symbols[ii][1:], mec='k',  label=folders[ii][10:])
#
# ox = np.arange(0, 1, 0.001)
# oy = 1 - ox
# plt.plot(ox, oy, 'k')
# plt.legend(loc=1, prop={'size': 9})
# plt.ylim((-.1, 1.1))
# plt.xlim((-.1, 1.1))
# plt.show()
