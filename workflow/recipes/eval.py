import tensorflow as tf 

class MeanIouCustom(tf.keras.metrics.MeanIoU):
  def __init__(self,num_classes,**kwargs):
    super().__init__(num_classes=num_classes,**kwargs)
  def update_state(self,y_true,y_pred,sample_weight=None):
    y_pred=tf.argmax(y_pred,axis=-1)
    return super().update_state(y_true,y_pred,sample_weight)