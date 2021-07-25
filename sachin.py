import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(r'C:\Users\ASUS\Desktop\sachinproject\brain tumor\Tumour\archive\Train', 
                                                 target_size = (256, 256),
                                                 batch_size = 32,
                                                 class_mode = 'binary') 

test_set = test_datagen.flow_from_directory(r'C:\Users\ASUS\Desktop\sachinproject\brain tumor\Tumour\archive\Test',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'binary')

cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=[256, 256, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid'))


cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=75, activation='relu'))

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) 

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])

j=cnn.fit_generator(training_set,
                  steps_per_epoch = 5,     
                  epochs = 5,
                  validation_data = test_set, 
                  validation_steps = 2) 


def AccuracyGraph():
    history=j
    epochs=5
    from matplotlib import pyplot as plt
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss=history.history['loss']
    val_loss=history.history['val_loss']
    epochs_range = range(epochs)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.show()
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    
AccuracyGraph()



def predict(d,p):
    import numpy as np
    from matplotlib import pyplot as plt
    from keras.preprocessing import image
    try:
        p=((d+'/')+p)
        test_image = image.load_img(p, target_size = (256, 256))
        plt.imshow(test_image)
        plt.title('Test Brain Image'), plt.xticks([]), plt.yticks([])
        plt.show()
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = cnn.predict(test_image)
        if result[0][0] == 1:
            prediction = '\n\n\t\t\tYou Are Suffering From Tumour'
        else:
            prediction = '\n\n\t\t\tYou Are Not Suffering From Tumour'
        return(prediction)
    except FileNotFoundError as m:
        print(m)

    
