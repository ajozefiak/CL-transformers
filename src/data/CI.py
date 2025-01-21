import numpy as np
import jax
import jax.random as jr
import jax.numpy as jnp
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_Imagenet32(dir):
    dicts = []
    for i in range(10):
        file = f'{dir}train_data_batch_{i+1}'
        dicts.append(unpickle(file))
    
    # Extract X_train
    list_of_data_arrays = [dicts[i]['data'] for i in range(10)]
    X_train = np.concatenate(list_of_data_arrays, axis=0)
    del list_of_data_arrays

    # Assuming X_train is your (N, 3072) numpy array
    N = X_train.shape[0]

    # Reshape X_train to (N, 32, 32, 3)
    X_train = X_train.reshape(N, 3, 32, 32).transpose(0, 2, 3, 1)

    # Extract y_train, subtracting off 1 so that labels range from 0 to 999
    list_of_labels_arrays = [np.array(dicts[i]['labels']) for i in range(10)]
    y_train = np.concatenate(list_of_labels_arrays, axis=0) - 1

    del dicts, list_of_labels_arrays

    path_val = f'{dir}val_data'
    val_dict = unpickle(path_val)

    X_test = val_dict['data']
    N_test = X_test.shape[0]
    X_test = X_test.reshape(N_test, 3, 32, 32).transpose(0, 2, 3, 1)

    # subtract 1 from y_test so that labels are between 0 and 999
    y_test = np.array(val_dict['labels']) - 1

    return X_train, y_train, X_test, y_test

# Use to permute order of images, at the beginning of an experiment
# so that we extract a random subset of 600 images of each class,
# and with each epoch so that the order of minibatches is always shuffled
def permute_image_order(X_train, y_train, key):
    permutation = jr.permutation(key, X_train.shape[0])
    return X_train[permutation], y_train[permutation]

# Next task always takes the first 600 images of each class
def get_next_task_CI32(X_train, y_train, X_test, y_test, task, family_0, family_1, images_per_class):
    class_0 = family_0[task]
    class_1 = family_1[task]

    X_train_0 = X_train[y_train == class_0][0:images_per_class]
    X_train_1 = X_train[y_train == class_1][0:images_per_class]
    X_train_ = jnp.concatenate((X_train_0, X_train_1), axis=0) / 255.0

    y_train_0 = y_train[y_train == class_0][0:images_per_class]
    y_train_1 = y_train[y_train == class_1][0:images_per_class]
    y_train_ = jnp.concatenate((y_train_0, y_train_1), axis=0)
    y_train_ = jnp.where(y_train_ == class_0, 0, 1)

    X_test_0 = X_test[y_test == class_0]
    X_test_1 = X_test[y_test == class_1]
    X_test_ = jnp.concatenate((X_test_0, X_test_1), axis=0) / 255.0

    y_test_0 = y_test[y_test == class_0]
    y_test_1 = y_test[y_test == class_1]
    y_test_ = jnp.concatenate((y_test_0, y_test_1), axis=0)
    y_test_ = jnp.where(y_test_ == class_0, 0, 1)

    return X_train_, y_train_, X_test_, y_test_