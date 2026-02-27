import tensorflow as tf
import numpy as np
loaded_model=tf.keras.models.load_model('models/brain_seg_v2.h5')
#/Users/johanneshounton/Space/models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize


def process_img(image_path):
    image_raw=tf.io.read_file(image_path)
    image=tf.image.decode_jpeg(image_raw,channels=3)
    image=tf.image.resize(image,[128,128])
    image=tf.expand_dims(image,axis=0)
    image=tf.cast(image,tf.float32)/255
    #return image

    #img=process_img(image_path)
    prediction=loaded_model.predict(image)
    prediction=np.argmax(prediction,axis=3)
    prediction=prediction[0,:,:]
    prediction=resize(prediction,(512,512),order=0,anti_aliasing=False,preserve_range=True)
    #prediction=prediction.astype('uint8')
    #prediction=np.array(Image.fromarray(prediction).resize((512,512)))
    array=(prediction*255).astype(np.uint8)
    Image.fromarray(array).save("./assets/test.jpg")
    return Image.fromarray(array)
process_img("assets/IMG-0002-00182.jpg")

