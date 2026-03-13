import tensorflow as tf
#Encoder
def encoder_block(inputs,num_filters):
  x=tf.keras.layers.Conv2D(num_filters,3,padding='same')(inputs)
  x=tf.keras.layers.Activation('relu')(x)
  x=tf.keras.layers.Conv2D(num_filters,3,padding='same')(x)
  x=tf.keras.layers.Activation('relu')(x)
  x=tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
  return x

