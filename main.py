# https://www.kaggle.com/datasets/nih-chest-xrays/data/data
# kod z: https://www.kaggle.com/code/adamjgoren/nih-chest-x-ray-multi-classification
import pandas as pd
from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # additional plotting functionality
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from sklearn.metrics import roc_curve, auc

# load data
xray_data = pd.read_csv('../Data_Entry_2017.csv')

# see how many observations there are
num_obs = len(xray_data)
print('Number of observations: ', num_obs)

# examine the raw data before performing pre-processing
xray_data.head(5)  # view first 5 rows
# xray_data.sample(5) # view 5 randomly sampled rows

my_glob = glob('../images*/images/*.png')
print('Number of Observations: ', len(my_glob))
# check to make sure I've captured every pathway, should equal 112,120

# Map the image paths onto xray_data
full_img_paths = {os.path.basename(x): x for x in my_glob}
xray_data['full_path'] = xray_data['Image Index'].map(full_img_paths.get)

# Q: how many unique labels are there? A: many (836) because of co-occurence
# Note: co-occurence will turn out to be a real pain to deal with later,
# but there are several techniques that help us work with it successfully
num_unique_labels = xray_data['Finding Labels'].nunique()
print('Number of unique labels:', num_unique_labels)

# let's look at the label distribution to better plan our next step
count_per_unique_label = xray_data['Finding Labels'].value_counts()  # get frequency counts per label
df_count_per_unique_label = count_per_unique_label.to_frame()
# convert series to dataframe for plotting purposes

print(df_count_per_unique_label)  # view tabular results
#  (sns.barplot(x=df_count_per_unique_label.index[:20], y="Finding Labels", data=df_count_per_unique_label[:20], color="green")
#  , plt.xticks(rotation=90))  # visualize results graphically

# define dummy labels for one hot encoding - simplifying to 14 primary classes (excl. No Finding)
dummy_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema', 'Emphysema', 'Fibrosis',
                'Effusion', 'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia']
# taken from paper

# One Hot Encoding of Finding Labels to dummy_labels
for label in dummy_labels:
    xray_data[label] = xray_data['Finding Labels'].map(lambda result: 1.0 if label in result else 0)
print(xray_data.head(20))  # check the data, looking good

# now, let's see how many cases present for each of our 14 clean classes (which excl. 'No Finding')
clean_labels = xray_data[dummy_labels].sum().sort_values(ascending=False)  # get sorted value_count for clean labels
print(clean_labels)  # view tabular results

# plot cases using seaborn barchart
clean_labels_df = clean_labels.to_frame()  # convert to dataframe for plotting purposes
sns.barplot(x=clean_labels_df.index[::], y=0, data=clean_labels_df[::], color="green"), plt.xticks(rotation=90)
# visualize results graphically

# create vector as ground-truth, will use as actuals to compare against our predictions later
xray_data['target_vector'] = xray_data.apply(lambda target: [target[dummy_labels].values], 1).map(
    lambda target: target[0])

train_set, test_set = train_test_split(xray_data, test_size=0.2, random_state=1993)

# quick check to see that the training and test set were split properly
print('training set - # of observations: ', len(train_set))
print('test set - # of observations): ', len(test_set))
print('prior, full data set - # of observations): ', len(xray_data))

# Create ImageDataGenerator, to perform significant image augmentation
# Utilizing most of the parameter options to make the image data even more robust
data_gen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)


# This Flow_from function is actually based on the default function from Keras '.flow_from_dataframe', but is more flexible
# Base function reference: https://keras.io/preprocessing/image/
# Specific notes re function: https://github.com/keras-team/keras/issues/5152

def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir,
                                              class_mode='sparse',
                                              **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = ''  # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


# Can use flow_from_dataframe() for training and validation - simply pass arguments through to function parameters
# Credit: Code adapted from Kevin Mader - Simple XRay CNN on 12/09/18
# https://www.kaggle.com/kmader/train-simple-xray-cnn

image_size = (128, 128)  # image re-sizing target
train_gen = flow_from_dataframe(data_gen, train_set, path_col='full_path', y_col='target_vector',
                                target_size=image_size, color_mode='grayscale', batch_size=32)
valid_gen = flow_from_dataframe(data_gen, test_set, path_col='full_path', y_col='target_vector', target_size=image_size,
                                color_mode='grayscale', batch_size=128)

# define test sets - here are problems
test_X, test_Y = next(
    flow_from_dataframe(data_gen, test_set, path_col='full_path', y_col='target_vector', target_size=image_size,
                        color_mode='grayscale',
                        batch_size=2048))

# Create CNN model
# Will use a combination of convolutional, max pooling, and dropout layers for this purpose
model = Sequential()

model.add(Conv2D(filters=8, kernel_size=3, padding='same', activation='relu', input_shape=test_X.shape[1:]))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=3))
model.add(Dropout(0.2))

# add in fully connected dense layers to model, then output classifiction probabilities using a softmax activation function
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(len(dummy_labels), activation='softmax'))

# compile model, run summary
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit_generator(generator=train_gen, steps_per_epoch=20, epochs=1, validation_data=(test_X, test_Y))

# Make prediction based on our fitted model
quick_model_predictions = model.predict(test_X, batch_size=64, verbose=1)

# create plot
fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
for (i, label) in enumerate(dummy_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:, i].astype(int), quick_model_predictions[:, i])
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (label, auc(fpr, tpr)))

# Set labels for plot
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('quick_trained_model.png')

## See previous code snippets for all references

# Run a longer, more detailed model
model.fit_generator(generator=train_gen, steps_per_epoch=50, epochs=5, validation_data=(test_X, test_Y))

# Make prediction based on our fitted model
deep_model_predictions = model.predict(test_X, batch_size=64, verbose=1)

# create plot
fig, c_ax = plt.subplots(1, 1, figsize=(9, 9))
for (i, label) in enumerate(dummy_labels):
    fpr, tpr, thresholds = roc_curve(test_Y[:, i].astype(int), deep_model_predictions[:, i])
    c_ax.plot(fpr, tpr, label='%s (AUC:%0.2f)' % (label, auc(fpr, tpr)))

# Set labels for plot
c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
fig.savefig('deep_trained_model.png')
