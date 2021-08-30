import tensorflow as tf

def main():
    
    #* load datasets
    test_dataset = tf.data.experimental.load('test_tf_dataset')
    
    model = tf.keras.models.load_model('tf_model')

    results = model.evaluate(test_dataset)
    print(f"RESULTS\n{results}")

    metrics = ['loss', 'accuracy', 'tp', 'fp', 'tn', 'fn', 'precision', 'recall']

    with open('test_results.txt', 'w') as f:
        for metric, result in zip(metrics, results):
            f.write(f'{metric}: {result}\n')

if __name__ == "__main__":
    main()
