import tensorflow as tf

def main():
    
    #* load datasets
    test_dataset = tf.data.experimental.load('test_tf_dataset')
    
    model = tf.keras.models.load_model('tf_model')

    results = model.evaluate(test_dataset)
    print(f"RESULTS\n{results}")


if __name__ == "__main__":
    main()