import pickle
import numpy as np
import tensorflow as tf


def write_testcase(file, testcase):

    filename = testcase['filename']
    y_pred = testcase['y_pred']
    label = testcase['label']


    with open(file, 'a') as f:

        for fname in filename:
            f.write(str(fname) + '\n')
                   
        f.write('y_pred: ' + str(y_pred) + '\n')
        f.write('label: ' + str(label) + '\n')

        f.write('--------------------------------------------' + '\n')



def main():

    #* load num_vec_to_file_dict
    with open('pickle/mozilla_num_vec_to_file_dict.pickle', 'rb') as f:
        num_vec_to_file_dict = pickle.load(f)
    
    #* load dataset
    test_dataset = tf.data.experimental.load('mozilla_tf_dataset')

    #* load model
    model = tf.keras.models.load_model('tf_model')

    #* evaluate
    for num_vecs, labels in test_dataset:

        y_preds = model.predict(num_vecs)
        y_preds = y_preds.flatten()

        labels = labels.numpy()

        batch_size = np.shape(labels)[0]

        errors = abs(y_preds - labels)

        for i in range(batch_size):

            y_pred = y_preds[i]
            label = labels[i]
            error = errors[i]

            num_vec = num_vecs[i]
            num_vec = num_vec.numpy()

            filename = num_vec_to_file_dict.get(tuple(num_vec))

            if not filename:
                continue
            
            testcase = {
                'filename': filename,
                'y_pred': y_pred,
                'label': label
            }

            #* mislabelled
            if error > 0.5:
                if label == 1:
                    #* false negative
                    write_testcase('mozilla_results/fn.txt', testcase)
                else:
                    #* false positive
                    write_testcase('mozilla_results/fp.txt', testcase)
            
            #* correctly labelled
            else:
                if label == 1:
                    #* true positive
                    write_testcase('mozilla_results/tp.txt', testcase)
                else:
                    #* true negative
                    write_testcase('mozilla_results/tn.txt', testcase)


    

if __name__ == "__main__":
    main()