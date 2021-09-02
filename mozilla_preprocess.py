import pickle
import numpy as np
import tensorflow as tf

from preprocess import *

def pad_num_vec_dict(num_vec_dict, vec_length):

    for fname, (vec, label) in num_vec_dict.items():
        np_vec = np.asarray(vec)
        np_vec.resize((vec_length,))

        num_vec_dict[fname] = (np_vec, label)
    
    return num_vec_dict


def get_num_vec_to_file_dict(num_vec_dict):

    # num_vec_to_file_dict: num_vec -> filename
    # NB: filename is a list in case multiple filenames have the same numerical vector 

    num_vec_to_file_dict = {}

    for fname, (num_vec, label) in num_vec_dict.items():
        num_vec_tuple = tuple(num_vec) # list can't be dict key, so convert num_vec to tuple (tuples can be dict keys)
        num_vec_to_file_dict[num_vec_tuple] = num_vec_to_file_dict.get(num_vec_tuple, []) + [fname]

    return num_vec_to_file_dict


def save_mozilla_dataset(num_vec_dict):
    vecs = []
    labels = []
    
    for vec, label in num_vec_dict.values():
        vecs.append(vec)
        labels.append(label)
    
    dataset = tf.data.Dataset.from_tensor_slices((vecs, labels))

    dataset = dataset.batch(32)

    tf.data.experimental.save(dataset, 'mozilla_tf_dataset')

def main():

    with open('pickle/token_to_num_map.pickle', 'rb') as f:
        token_to_num_map = pickle.load(f)

    with np.load('train_vars.npz') as train_vars:
        vocab_size = train_vars['vocab_size']
        longest_vec_length = train_vars['input_len']


    mozilla_token_vec_dict = get_token_vec_dict('mozilla.json')
    pickle_dict(mozilla_token_vec_dict, 'pickle/mozilla_token_vec_dict.pickle')
    
    mozilla_num_vec_dict = get_num_vec_dict(mozilla_token_vec_dict, token_to_num_map)
    mozilla_longest_vec_length = get_longest_vec_length([mozilla_num_vec_dict])

    if mozilla_longest_vec_length > longest_vec_length:
        print(f'mozilla longest: {mozilla_longest_vec_length} > dataset longest: {longest_vec_length}')
        longest_vec_length = mozilla_longest_vec_length
    
    mozilla_num_vec_dict = pad_num_vec_dict(mozilla_num_vec_dict, longest_vec_length)

    save_mozilla_dataset(mozilla_num_vec_dict)
    
    #* num_vec to file dict
    num_vec_to_file_dict = get_num_vec_to_file_dict(mozilla_num_vec_dict)
    pickle_dict(num_vec_to_file_dict, 'pickle/mozilla_num_vec_to_file_dict.pickle')

    save_train_vars(vocab_size, longest_vec_length)

if __name__ == "__main__":
    main()