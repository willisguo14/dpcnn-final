import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def main():

    with np.load('train_vars.npz') as train_vars:
        vocab_size = train_vars['vocab_size']
        input_len = train_vars['input_len']
    
    #* load datasets
    train_dataset = tf.data.experimental.load('train_tf_dataset')
    val_dataset = tf.data.experimental.load('val_tf_dataset')
    test_dataset = tf.data.experimental.load('test_tf_dataset')
    
    #* build model 
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=30, input_length=input_len),
        tf.keras.layers.Conv1D(filters=10, kernel_size=5, activation=tf.nn.relu, kernel_regularizer='l2'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=100, activation=tf.nn.relu, kernel_regularizer='l2'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
    ])

    print(model.summary())

    #* compile model
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
    ]
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=metrics
    )

    history = model.fit(train_dataset, epochs=15, validation_data=val_dataset)

    #* plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('accuracy.png')

    #* save model
    model.save('tf_model')

if __name__ == "__main__":
    main()
