import json
import pickle
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def print_json(data):
    print(json.dumps(data, indent=2))
def print_dict(my_dict):
    for key, value in my_dict.items():
        print(f"{key} => {value}\n")

def pickle_dict(d, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(d, f)

 
def get_token(node):

    tokens = []

    node_type = node["_label"]

    if node_type == "METHOD_PARAMETER_IN":
        tokens.append(node['typeFullName'])
        tokens.append(node['name'])

    elif node_type == "IDENTIFIER" or node_type == "LITERAL":
        if node['typeFullName'] in ['short', 'int', 'long', 'float', 'double', 'char', 'unsigned', 'signed', 'void' ,'wchar_t', 'size_t', 'bool']:
            tokens.append(node['typeFullName'])
        else:
            tokens.append('object')

    elif node_type == "CALL":
        tokens.append(node['name'])

    elif node_type == "METHOD":
        tokens.append('function')

    elif node_type == "CONTROL_STRUCTURE":
        tokens.append(node['controlStructureType'])

    elif node_type == "BLOCK":
        tokens.append(node['typeFullName'])

    return tokens

def get_label(fname):

    if "d2a" in fname:

        f1 = os.path.basename(os.path.normpath(fname))
        f2 = os.path.splitext(f1)[0]

        if f2.endswith('_1'):
            return 1
        else:
            return 0
    
    else: # NVD or SARD 

        if "VULN" in fname or "BAD" in fname:
            return 1
        else:
            return 0

def get_token_vec_dict(json_file):

    # token_vec_dic: key = filename, value = ([token_vec], label)
    
    token_vec_dict = {}
    
    with open(json_file, 'r') as f:
        ast = json.load(f)
    
    
    cur_file = ""
    cur_vec = []
    cur_label = -1

    for node in ast:
        
        #* found a node with "filename" properties = start of a new 'block' 
        if "filename" in node.keys():

            #* add previous 'block'
            if cur_file and cur_vec and cur_label != -1:
                token_vec_dict[cur_file] = (cur_vec, cur_label)

            #* update cur_file, cur_vec, cur_label for new 'block'
            cur_file = node["filename"]

            # unseen source code file
            if cur_file not in token_vec_dict.keys():
                cur_vec = []
                cur_label = get_label(cur_file)
            # source code file seen before
            else:
                cur_vec, cur_label = token_vec_dict[cur_file]
        
        #* get appropriate token
        tokens = get_token(node)

        #* only if this node is 'selected' 
        if tokens:
            for token in tokens:
                #* append token
                cur_vec.append(token)
    
    #* add last block after exiting for loop 
    if cur_file and cur_vec and cur_label != -1: 
        token_vec_dict[cur_file] = (cur_vec, cur_label)
    
    return token_vec_dict


def update_token_freq_dict(token_vec_dict, token_freq_dict):

    for fname, (token_vec, label) in token_vec_dict.items():

        for token in token_vec:
            token_freq_dict[token] = token_freq_dict.get(token, 0) + 1
    
    return token_freq_dict


def get_token_to_num_map(token_freq_dict):

    token_to_num_map = {}

    num = 1 

    for token, freq in token_freq_dict.items():
        
        if freq >= 3:
            token_to_num_map[token] = num
            num += 1
    
    vocab_size = num
    
    return token_to_num_map, vocab_size


def token_to_num_vec(token_vec, token_to_num_map):
    num_vec = [token_to_num_map.get(token, 0) for token in token_vec]
    return num_vec

def get_num_vec_dict(token_vec_dict, token_to_num_map):

    num_vec_dict = {}

    for fname, (token_vec, label) in token_vec_dict.items():
        num_vec = token_to_num_vec(token_vec, token_to_num_map)
        num_vec_dict[fname] = (num_vec, label)
    
    return num_vec_dict


def get_longest_vec_length(num_vec_dicts):

    longest_vec_length = -1
    
    for num_vec_dict in num_vec_dicts:
        for vec, _ in num_vec_dict.values():
            longest_vec_length = max(longest_vec_length, len(vec))
    
    return longest_vec_length



def save_dataset(num_vec_dict, vec_length, dataset_name):

    vecs = []
    labels = []
    
    for vec, label in num_vec_dict.values():
        np_vec = np.asarray(vec)
        np_vec.resize((vec_length,))
        vecs.append(np_vec)
        
        labels.append(label)
    
    dataset = tf.data.Dataset.from_tensor_slices((vecs, labels))

    dataset = dataset.batch(32)

    tf.data.experimental.save(dataset, dataset_name)


def save_datasets(train_num_vec_dict, val_num_vec_dict, test_num_vec_dict, longest_vec_length, vocab_size):

    save_dataset(train_num_vec_dict, longest_vec_length, 'train_tf_dataset')
    save_dataset(val_num_vec_dict, longest_vec_length, 'val_tf_dataset')
    save_dataset(test_num_vec_dict, longest_vec_length, 'test_tf_dataset')


def save_train_vars(vocab_size, input_len):
    np.savez('train_vars', vocab_size=vocab_size, input_len=input_len)

def main():
    train_token_vec_dict = get_token_vec_dict('ast/train.json')
    val_token_vec_dict = get_token_vec_dict('ast/val.json')
    test_token_vec_dict = get_token_vec_dict('ast/test.json')

    

    pickle_dict(train_token_vec_dict, 'pickle/train_token_vec_dict.pickle')
    pickle_dict(val_token_vec_dict, 'pickle/val_token_vec_dict.pickle')
    pickle_dict(test_token_vec_dict, 'pickle/test_token_vec_dict.pickle')


    print(len(train_token_vec_dict))
    print(len(val_token_vec_dict))
    print(len(test_token_vec_dict))

    token_freq_dict = {}
    token_freq_dict = update_token_freq_dict(train_token_vec_dict, token_freq_dict)
    token_freq_dict = update_token_freq_dict(val_token_vec_dict, token_freq_dict)
    
    token_to_num_map, vocab_size = get_token_to_num_map(token_freq_dict)

    pickle_dict(token_to_num_map, 'pickle/token_to_num_map.pickle')

    # print_dict(token_to_num_map)
    print(f"VOCAB SIZE: {vocab_size}")

    train_num_vec_dict = get_num_vec_dict(train_token_vec_dict, token_to_num_map)
    val_num_vec_dict = get_num_vec_dict(val_token_vec_dict, token_to_num_map)
    test_num_vec_dict = get_num_vec_dict(test_token_vec_dict, token_to_num_map)

    longest_vec_length = get_longest_vec_length([train_num_vec_dict, val_num_vec_dict])

    save_datasets([train_num_vec_dict, val_num_vec_dict, test_num_vec_dict], longest_vec_length, vocab_size)
    
    save_train_vars(vocab_size, longest_vec_length)

if __name__ == "__main__":
    main()
