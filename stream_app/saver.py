import tensorflow as tf 
loaded_model=tf.keras.models.load_model('E:/memoire/stream_app/src/models/brain_seg.h5')


loaded_model.save('src/models/brain_seg')