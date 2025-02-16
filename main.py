
import streamlit as st 
import pandas as pd 
import tensorflow as tf
import numpy as np

#Tensorflow Model Prediction
def model_prediction(test_image):
    model=tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])
    prediction=model.predict(input_arr)
    result_index=np.argmax(prediction) 
    return  result_index

#Sidebar
st.sidebar.title("Dashboard")
app_mode=st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition"])

#Home page
if (app_mode=="Home"):
    st.header("Plant Disease recognition system")
    image_path = "home_page.jpg"
    st.image(image_path,use_column_width=True)
    st.markdown("""Welcome to the Plant Disease Recognition System! 🌿🔍
    

#abput Page
elif(app_mode=="About"):
    st.header("About")    
    st.markdown("""
                #about Dataset\n
                This dataset is recreated using offline augmentation from the original dataset.\n
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.
                The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)


                 """)
 
#Prediction page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image=st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,use_column_width=True)
    #Predict Button
    if(st.button("Predict")):
        with st.balloons():
            st.write("Our prediction")
        result_index=model_prediction(test_image)

        #DEfine class
        class_name=['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

