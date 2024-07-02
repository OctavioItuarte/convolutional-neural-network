from tensorflow.keras import models
import matplotlib.pyplot as plt

def show_data(history, model, train_images, train_labels, test_images, test_labels):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.0, 1])
    plt.legend(loc='lower right')

    #model = models.load_model('data/my_model')

    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

    print(test_acc)