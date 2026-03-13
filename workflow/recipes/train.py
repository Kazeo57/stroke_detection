from dataprep.load_and_process import get_dataset
from src.simple_unet import unet_model
import tensorflow as tf 
from eval import MeanIouCustom
train_ds=get_dataset('.dataprep/Real_hypodensity-5/train')
valid_ds=get_dataset('.dataprep/Real_hypodensity-5/valid')


if __name__ == '__main__':
    model = unet_model(input_shape=(128, 128, 3), num_classes=5)
    model.summary()
    model.compile(optimizer="Adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=[MeanIouCustom(num_classes=5),tf.keras.metrics.SparseCategoricalAccuracy()])
    early_stopping=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)
    history=model.fit(train_ds,validation_data=valid_ds,epochs=60,callbacks=early_stopping)

