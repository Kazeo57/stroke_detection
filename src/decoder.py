import tensorflow as tf 
#Decoder
def decoder_block(inputs,skip_features,num_filters):
  x=tf.keras.layers.Conv2DTranspose(num_filters,(2,2),strides=2,padding='same')(inputs)
  skip_features=tf.keras.layers.Resizing(x.shape[1],x.shape[2])(skip_features)

  x=tf.keras.layers.Concatenate()([x,skip_features])

  x=tf.keras.layers.Conv2D(num_filters,3,padding='same')(x)
  x=tf.keras.layers.Activation('relu')(x)
  x=tf.keras.layers.Conv2D(num_filters,3,padding='same')(x)
  x=tf.keras.layers.Activation('relu')(x)
  return x