import streamlit as st
import pandas as pd
from io import StringIO

import os 
from PIL import Image 
from predictor import process 
import numpy as np
#def main():
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    img=Image.open(uploaded_file)
    img.save(uploaded_file.name)
    #os.path.join('',uploaded_file.name)
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #pred=predict(bytes)
    #st.write(bytes_data)
    col1,col2=st.columns(2)
    #st.image(bytes_data)
    processed_img=process(uploaded_file.name)
    colors={0:[0,0,0],
        1:[255,0,255],
        2:[255,255,0],
        3:[0,255,255],
        4:[0,0,255]}
    h,w=processed_img.shape 
    
    colored_img=np.zeros((h,w,3),dtype=np.uint8)
    original_img=np.array(Image.open(uploaded_file.name))
    

    for cls,color in colors.items():
        colored_img[processed_img==cls]=color
    img_to_show=((1-0.5)*original_img + 0.5*colored_img).astype(np.uint8)
    with col1:
        st.subheader("Scanner du patient")
        st.image(bytes_data)#,use_column_width=True)
    
    with col2:
        st.subheader("Image traitee")
        st.image(colored_img)
        #st.image(img_to_show)#,use_column_width=True)
    # To convert to a string based IO:
    #stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    #string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    #dataframe = pd.read_csv(uploaded_file)
    #st.write(dataframe)


    #if __name__=='__main__':
    #os.system('streamlit run app.py')
