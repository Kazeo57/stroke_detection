import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse 
import tensorflow as tf 
from dataprep.load_and_process import get_dataset 
from eval import MeanIouCustom 

from src.simple_unet import unet_model
from src.super_unet import pre_vgg_unet, pre_resnet_unet, pre_efficientnet_unet ,create_mobilenet_unet

ARCHITECTURES={
    "simple_unet":unet_model,
    "mobilenet_unet":create_mobilenet_unet,
    "vggnet_unet":pre_vgg_unet,
    "resnet_unet":pre_resnet_unet,
    "efficientnet_unet":pre_efficientnet_unet
}


def parse_args():
    parser=argparse.ArgumentParser(description="Brain Segmentation Training ...")
    parser.add_arguments("--arch",type=str,default="simpleunet",choices=ARCHITECTURES.keys(),help="Architecture")
    #input_shape=(256, 256, 3), num_classes=5
    parser.add_argument("--input_shape", type=int, nargs=2, default=[128, 128],
                        help="Dimensions de l'image (H W), ex: --input_shape 256 256")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Nombre de classes de segmentation")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Nombre max d'epochs")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience pour l'early stopping")
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="Optimiseur (adam, sgd, rmsprop...)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--train_dir", type=str, default=".dataprep/Real_hypodensity-5/train")
    parser.add_argument("--valid_dir", type=str, default=".dataprep/Real_hypodensity-5/valid")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Chemin pour sauvegarder le modèle entraîné (.keras)")

    return parser.parse_args()




if __name__ == '__main__':
    args=parse_args()

    train_ds=get_dataset(args.train_dir)
    valid_ds=get_dataset(args.valid_dir)

    model_fn = ARCHITECTURES[args.arch]#unet_model(input_shape=(128, 128, 3), num_classes=5)
    input_shape=(args.input_shape,3)
    model=model_fn(input_shape=input_shape,num_classes=args.num_classes)

    model.summary()

    optimizer=tf.keras.optimizers.get({
        "class_name": args.optimizer,
        "config":{"learning_rate":args.lr}
    })

    #model.compile(optimizer="Adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=[MeanIouCustom(num_classes=5),tf.keras.metrics.SparseCategoricalAccuracy()])
    model.compile(optimizer=optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=[MeanIouCustom(num_classes=5),tf.keras.metrics.SparseCategoricalAccuracy()])
    early_stopping=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,restore_best_weights=True)
    history=model.fit(train_ds,validation_data=valid_ds,epochs=60,callbacks=early_stopping)
