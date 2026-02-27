import tensorflow as tf 
import skimage

import numpy as np
#import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.segmentation import chan_vese
from skimage.util import img_as_float
from skimage.measure import  regionprops,label



loaded_model=tf.keras.models.load_model('E:/memoire/stream_app/src/models/brain_seg_v2.h5')


def process_img(image_path):
  image_raw=tf.io.read_file(image_path)
  image=tf.image.decode_jpeg(image_raw,channels=3)
  image=tf.image.resize(image,[128,128])
  image=tf.expand_dims(image,axis=0)
  image=tf.cast(image,tf.float32)/255
  return image

def process(path):
  img=process_img(path)
  prediction=loaded_model.predict(img)
  prediction=np.argmax(prediction,axis=3)
  prediction=prediction[0,:,:]
  prediction=resize(prediction,(512,512),order=0,anti_aliasing=False,preserve_range=True)
  #prediction=prediction.astype('uint8')
  #prediction=np.array(Image.fromarray(prediction).resize((512,512)))
  #plt.imshow(prediction)
  labels_id=[i for i in np.unique(prediction) if i!=0]
  regions=[]
  for label in labels_id:
    region_mask=(prediction==label).astype(np.uint8)
    regions.append(region_mask)
  #return regions
  return (prediction,regions)


def find_hypodensity(target,mask):#,all_mask):
  """
  target : constitue la zone de l'image originel où il faut rechercher l'hypodesnité
  mask :c'est le masque correspondant à cette zone
  -->
  if_hypodensity :Vérifie si cette zone contient une hypodesnité ou non
  """
  if_hypodensity=False
  #Chan vese segmentation uniquement sur les pixels de la régions

  ###A revoir
  image=img_as_float(target)
  image_masked=np.where(mask,image,np.nan)
  neutral_value=np.nanmean(image_masked)
  image_filled=np.where(np.isnan(image_masked),neutral_value,image_masked)
  cv=chan_vese(
      image_filled,#target,
      mu=0.25,
      lambda1=1,
      lambda2=1,
      tol=1e-3,
      max_num_iter= 200,
      dt=0.5,
      init_level_set='checkerboard',#'disk',#
      extended_output=False
  )

  cv_masked=np.where(mask,cv,0)
  #plt.imshow(cv_masked,cmap='gray')
  #plt.title("Segmentation")
  #plt.show()
  cv_bool=cv_masked.astype(bool)
  hypo=np.logical_and(mask,~cv_bool)

  hypo_clean=remove_small_objects(hypo,min_size=100)
  hypo_clean=remove_small_holes(hypo_clean,area_threshold=100)
  #plt.imshow(hypo_clean,cmap='gray')
  #plt.title("Segmentation Epurée")
  #plt.show()
  ###End
  hypo_regs=regionprops(label(hypo_clean))
  print("Num potential hypodensity layer",len(hypo_regs))
  #available_area=[]
  hypo_img=None
  #all_mask=np.zeros(,dtype=np.uint8)
  mask_img = np.zeros(mask.shape, dtype=np.uint8)


  for i,hyp_reg in enumerate(hypo_regs):
    print(f"hypo :{i} ::> Area :{hyp_reg.area}")
    #available_area.append(hyp_reg.area)
    if  (0,1) in np.unique(hypo_clean) and hyp_reg.area>=5000 and hyp_reg.area/hyp_reg.convex_area>=0.5:
        print("========================True!!!==========================")
        if_hypodensity=True
        # récupération des pixels de la région
        coords = hyp_reg.coords  # tableau Nx2 → [ [row, col], ... ]

        # remplir le masque final
        mask_img[coords[:, 0], coords[:, 1]] = 255
        hypo_img=mask_img

  #if hypo_img:
  return hypo_img
 
  
  



class HypoTron:
  def __init__(self,zone_model,hypo_density_factor):
    pass


  #def __call__():
  #return prediction"""

