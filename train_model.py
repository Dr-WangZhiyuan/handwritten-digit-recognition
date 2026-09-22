import tensorflow as tf
import numpy as np
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, epochs=10, batch_size=128,
          validation_data=(x_test, y_test), verbose=1)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f'\nTest accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

predictions = model.predict(x_test[:20], verbose=0)
print('\nSample predictions vs actual:')
for i in range(20):
    pred = np.argmax(predictions[i])
    actual = y_test[i]
    conf = predictions[i][pred] * 100
    status = 'OK' if pred == actual else 'WRONG'
    print(f'  #{i}: predicted={pred} actual={actual} conf={conf:.1f}% {status}')

save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist_cnn.keras')
model.save(save_path)
print(f'\nModel saved to {save_path}')
