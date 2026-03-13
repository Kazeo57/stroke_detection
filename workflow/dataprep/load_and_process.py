from roboflow import Roboflow
from dotenv import load_dotenv 
from os import getenv


load_dotenv("local_env")

robloflow_api=getenv("ROBOFLOW_API")
rf = Roboflow(api_key=robloflow_api)
project = rf.workspace("stagevisionparordinateur").project("real_hypodensity")
version = project.version(5)
dataset = version.download("png-mask-semantic")


import tensorflow as tf
import os
#In my code image are already resized otherwise you must add it in process step
def load_image_and_mask(image_path,mask_path):
  image=tf.io.read_file(image_path)
  image=tf.cast(tf.image.decode_jpeg(image,channels=3),tf.float32)
  image=tf.image.resize(image,[128,128],method="nearest")/255.0
  #eVENTUAL RESIZE STEP

  mask=tf.io.read_file(mask_path)
  mask=tf.image.decode_png(mask,channels=1)
  #eVENTUAL RESIZE STEP
  mask=tf.image.resize(mask,[128,128],method="nearest")
  mask=tf.cast(mask,tf.uint8)

  return image,mask

def get_dataset(dir):
  image_files=sorted([f for f in os.listdir(dir) if f.endswith('.jpg')])
  images_paths=[os.path.join(dir,fname) for fname in image_files]
  mask_files=sorted([f for f in os.listdir(dir) if f.endswith('_mask.png')])
  mask_paths=[os.path.join(dir,fname) for fname in mask_files]

  ds=tf.data.Dataset.from_tensor_slices((images_paths,mask_paths))
  ds=ds.map(load_image_and_mask,num_parallel_calls=tf.data.AUTOTUNE)
  ds=ds.batch(16).prefetch(tf.data.AUTOTUNE)
  return ds
train_ds=get_dataset('./Real_hypodensity-5/train')
valid_ds=get_dataset('./Real_hypodensity-5/valid')
test_ds=get_dataset('./Real_hypodensity-5/test')
