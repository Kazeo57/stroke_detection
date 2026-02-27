import streamlit as st
import numpy as np
from PIL import Image
from predictor import process, find_hypodensity   

st.title("Stroke Processing")

# --- Upload de fichier ---
uploaded_file = st.file_uploader("Upload une image", type=["png", "jpg", "jpeg"])

# --- Choix du traitement ---
options = ["Territoires vasculaires", "Hypodensité", "Traitement mixte"]
chosen_processing = st.selectbox("Choisir un type de traitement", options)

# --- Affichage de l'image uploadée ---
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_gray=Image.open(uploaded_file).convert('L')
    img_array = np.array(img_gray)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image originale")
        st.image(img, use_container_width=True)

    # Bouton pour lancer le traitement
    if st.button("Process"):
        # Enregistrer temporairement si ton modèle nécessite un chemin
        temp_path = f"temp_{uploaded_file.name}"
        img.save(temp_path)
        vascular_segmentation,regions=process(temp_path)
        original_img=np.array(img)
        # --- APPELER LE BON TRAITEMENT SELON LE CHOIX ---
        if chosen_processing == "Territoires vasculaires":
            processed_img = vascular_segmentation

            # Construction de l'image couleur
            colors = {
                0: [0, 0, 0],
                1: [255, 0, 255],
                2: [255, 255, 0],
                3: [0, 255, 255],
                4: [0, 0, 255],
            }

            h, w = processed_img.shape
            colored_img = np.zeros((h, w, 3), dtype=np.uint8)

            for cls, color in colors.items():
                colored_img[processed_img == cls] = color

            img_to_show = ((1 - 0.5) * original_img + 0.5 * colored_img).astype(np.uint8)

            with col2:
                st.subheader("Image traitée")
                st.image(img_to_show, use_container_width=True)

        elif chosen_processing=="Hypodensité":
            with col2:
                #hypo_img=img

                st.subheader("Image traitée")
                #hypo_presence=[]
                #for i in range(len(regions)):
                hypo_img=find_hypodensity(img_array*regions[2],regions[2])
                    #if hypo_img is None: 
                        #pass 
                    #else:
                #if hypo_img!=img:

                st.image(hypo_img,use_container_width=True)
                #else:
                #st.write("⚠️ Ce traitement est insdisponible.")

        elif chosen_processing == "Traitement mixte":
            with col2:
                st.subheader("Image mixte")

                # --- 1) Territoires vasculaires ---
                processed_img = vascular_segmentation

                colors = {
                    0: [0, 0, 0],
                    1: [255, 0, 255],
                    2: [255, 255, 0],
                    3: [0, 255, 255],
                    4: [0, 0, 255],
                }

                h, w = processed_img.shape
                colored_img = np.zeros((h, w, 3), dtype=np.uint8)

                for cls, color in colors.items():
                    colored_img[processed_img == cls] = color

                # Fusion originale + territoires
                vascular_overlay = ((1 - 0.5) * original_img + 0.5 * colored_img).astype(np.uint8)

                # --- 2) Hypodensité ---
                hypo_mask = find_hypodensity(img_array * regions[2], regions[2])
                # hypo_mask = 2D image 0/255

                # Convertir le masque en RGB rouge translucide
                red_layer = np.zeros_like(vascular_overlay)
                red_layer[:, :, 0] = hypo_mask  # canal rouge seulement

                # Fusion hypodensité + image vasculaire
                alpha = 0.5  # transparence du rouge
                mixed_result = vascular_overlay.copy()
                mask_bool = hypo_mask.astype(bool)

                mixed_result[mask_bool] = (
                    (1 - alpha) * vascular_overlay[mask_bool] +
                    alpha * red_layer[mask_bool]
                ).astype(np.uint8)

                # --- Affichage final ---
                st.image(mixed_result, use_container_width=True)

