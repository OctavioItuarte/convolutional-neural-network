import tensorflow as tf
from tensorflow.keras import optimizers
from models.cnn_model import create_model
from utils.data_preprocessing import load_data, preprocess_data
from utils.data_eval import show_data

def train(data_dir):
    (train_images, train_labels), (test_images, test_labels) = load_data(data_dir)
    train_images, test_images = preprocess_data(train_images, test_images)
    model = create_model()
    
    optm = optimizers.Adam(learning_rate=0.00001)
    model.compile(optimizer=optm, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

    history = model.fit(train_images, train_labels, batch_size=128, epochs=10, validation_data=(test_images, test_labels))
    
    #model.save('data/my_model', save_format='tf')
    #model.save('/data/my_model.h5')
    show_data(history, model, train_images, train_labels, test_images, test_labels)

    return history

if __name__ == "__main__":
    train()