model.add(tf.keras.layers.Flatten(input_shape=(image_pixels, image_pixels)))
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.1))
model.add(tf.keras.layers.Dense(units=AMOUNT_OF_LETTER_CATEGORIES, activation='softmax'))


#@title Hyperparameters
learning_rate = 0.2
epochs = 55
batch_size = 64
validation_split = 0.2
