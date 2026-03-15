import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D ,MaxPooling2D ,UpSampling2D,concatenate,Conv2DTranspose,BatchNormalization,Dropout ,Activation
from decoder import decoder_block

def create_mobilenet_unet(input_shape,num_classes):
  assert num_classes>1
  base_model=tf.keras.applications.MobileNetV2(input_shape=input_shape,include_top=False)

  layer_names=[
      'block_1_expand_relu', # 4096
      'block_3_expand_relu', #1024
      'block_6_expand_relu', #256
      'block_13_expand_relu', #64
      'block_16_project',  #16
  ]

  base_model_outputs=[base_model.get_layer(name).output for name in layer_names]

  down_stack=tf.keras.Model(inputs=base_model.input,outputs=base_model_outputs)

  down_stack.trainable=False



  #up_stack
  class UP_STACK():
    def __init__(self,bridge,p1,p2,p3,p4):
      #bridge=conv_block(p4,n_filters=1024)
      self.u4=decoder_block(bridge,p4,512)
      self.u3=decoder_block(self.u4,p3,256)
      self. u2=decoder_block(self.u3,p2,128)
      self.u1=decoder_block(self.u2,p1,64)


  inputs=tf.keras.layers.Input(shape=input_shape)

  skips=down_stack(inputs)
  p1=skips[0]
  p2=skips[1]
  p3=skips[-3]
  p4=skips[-2]
  bridge=skips[-1]
  print(f"P1 :{p1} P2 :{p2} P3 :{p3} P4 :{p4}")
  up_stack=UP_STACK(bridge,p1,p2,p3,p4)
  #p4=skips[-1]
  print("U1 ",up_stack.u1)
  #outputs=Conv2DTranspose(filters=5,kernel_size=(1,1),strides=(2,2),activation='softmax')(up_stack.u1)
  outputs=Conv2D(filters=num_classes,kernel_size=(1,1),activation='softmax')(up_stack.u1)
  #outputs=tf.keras.layers.Resizing(128,128,interpolation='nearest')(outputs)
  outputs=tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(outputs)
  return Model(inputs=[inputs],outputs=[outputs])


def pre_vgg_unet(input_shape,num_classes):
  base_vgg_model=tf.keras.applications.VGG16(input_shape=input_shape,include_top=False)

  vgg_layer_names=[
      'block1_conv2', # 4096
      'block2_conv2', #1024
      'block3_conv3', #256
      'block4_conv3', #64
      'block5_conv3',  #16
  ]

  base_model_outputs=[base_vgg_model.get_layer(name).output for name in vgg_layer_names]

  down_stack=tf.keras.Model(inputs=base_vgg_model.input,outputs=base_model_outputs)

  down_stack.trainable=False



  #up_stack
  class UP_STACK():
    def __init__(self,bridge,p1,p2,p3,p4):
      #bridge=conv_block(p4,n_filters=1024)
      self.u4=decoder_block(bridge,p4,512)
      self.u3=decoder_block(self.u4,p3,256)
      self. u2=decoder_block(self.u3,p2,128)
      self.u1=decoder_block(self.u2,p1,64)


  inputs=tf.keras.layers.Input(shape=[128,128,3])

  skips=down_stack(inputs)
  p1=skips[0]
  p2=skips[1]
  p3=skips[-3]
  p4=skips[-2]
  bridge=skips[-1]
  print(f"P1 :{p1} P2 :{p2} P3 :{p3} P4 :{p4}")
  up_stack=UP_STACK(bridge,p1,p2,p3,p4)
  #p4=skips[-1]
  print("U1 ",up_stack.u1)
  #outputs=Conv2DTranspose(filters=5,kernel_size=(1,1),strides=(2,2),activation='softmax')(up_stack.u1)
  outputs=Conv2D(filters=num_classes,kernel_size=(1,1),activation='softmax')(up_stack.u1)
  #outputs=tf.keras.layers.Resizing(128,128,interpolation='nearest')(outputs)
  #outputs=tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(outputs)
  return Model(inputs=[inputs],outputs=[outputs])



def pre_resnet_unet(input_shape,num_classes):
  assert num_classes>1
  base_resnet_model=tf.keras.applications.ResNet50(input_shape=input_shape,include_top=False)

  resnet_layer_names = [
    'conv1_relu',
    'conv2_block3_out',
    'conv3_block4_out',
    'conv4_block6_out',
    'conv5_block3_out',
   ]

  base_model_outputs=[base_resnet_model.get_layer(name).output for name in resnet_layer_names]

  down_stack=tf.keras.Model(inputs=base_resnet_model.input,outputs=base_model_outputs)

  down_stack.trainable=False



  #up_stack
  class UP_STACK():
    def __init__(self,bridge,p1,p2,p3,p4):
      #bridge=conv_block(p4,n_filters=1024)
      self.u4=decoder_block(bridge,p4,512)
      self.u3=decoder_block(self.u4,p3,256)
      self. u2=decoder_block(self.u3,p2,128)
      self.u1=decoder_block(self.u2,p1,64)


  inputs=tf.keras.layers.Input(shape=[128,128,3])

  skips=down_stack(inputs)
  p1=skips[0]
  p2=skips[1]
  p3=skips[-3]
  p4=skips[-2]
  bridge=skips[-1]
  print(f"P1 :{p1} P2 :{p2} P3 :{p3} P4 :{p4}")
  up_stack=UP_STACK(bridge,p1,p2,p3,p4)
  #p4=skips[-1]
  print("U1 ",up_stack.u1)
  #outputs=Conv2DTranspose(filters=5,kernel_size=(1,1),strides=(2,2),activation='softmax')(up_stack.u1)
  outputs=Conv2D(filters=num_classes,kernel_size=(1,1),activation='softmax')(up_stack.u1)
  #outputs=tf.keras.layers.Resizing(128,128,interpolation='nearest')(outputs)
  outputs=tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(outputs)
  return Model(inputs=[inputs],outputs=[outputs])




efficient_layer_names = [
    'block2a_expand_activation',
    'block3a_expand_activation',
    'block4a_expand_activation',
    'block6a_expand_activation',
    'top_activation',
]

def pre_efficientnet_unet(input_shape,num_classes):
  assert num_classes>1
  base_efficient_model=tf.keras.applications.EfficientNetB0(input_shape=input_shape,include_top=False)

  efficient_layer_names = [
    'block2a_expand_activation',
    'block3a_expand_activation',
    'block4a_expand_activation',
    'block6a_expand_activation',
    'top_activation',
  ]

  base_model_outputs=[base_efficient_model.get_layer(name).output for name in efficient_layer_names]

  down_stack=tf.keras.Model(inputs=base_efficient_model.input,outputs=base_model_outputs)

  down_stack.trainable=False



  #up_stack
  class UP_STACK():
    def __init__(self,bridge,p1,p2,p3,p4):
      #bridge=conv_block(p4,n_filters=1024)
      self.u4=decoder_block(bridge,p4,512)
      self.u3=decoder_block(self.u4,p3,256)
      self. u2=decoder_block(self.u3,p2,128)
      self.u1=decoder_block(self.u2,p1,64)


  inputs=tf.keras.layers.Input(shape=[128,128,3])

  skips=down_stack(inputs)
  p1=skips[0]
  p2=skips[1]
  p3=skips[-3]
  p4=skips[-2]
  bridge=skips[-1]
  print(f"P1 :{p1} P2 :{p2} P3 :{p3} P4 :{p4}")
  up_stack=UP_STACK(bridge,p1,p2,p3,p4)
  #p4=skips[-1]
  print("U1 ",up_stack.u1)
  #outputs=Conv2DTranspose(filters=5,kernel_size=(1,1),strides=(2,2),activation='softmax')(up_stack.u1)
  outputs=Conv2D(filters=num_classes,kernel_size=(1,1),activation='softmax')(up_stack.u1)
  #outputs=tf.keras.layers.Resizing(128,128,interpolation='nearest')(outputs)
  outputs=tf.keras.layers.UpSampling2D(size=(2,2),interpolation='nearest')(outputs)
  return Model(inputs=[inputs],outputs=[outputs])









