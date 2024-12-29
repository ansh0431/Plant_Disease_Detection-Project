import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "Home_page.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Detection System! 🌿🔍
    
    Our mission is to help in identifying cotton crop diseases efficiently. Upload an image, and our system will analyze it to detect any signs of diseases. Together, let's protect our cotton crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our cotton Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
    This dataset consists of about 7.8K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
    
    #### Content
    1. Train (6251 images)
    2. Validation (1563 images)


    #### Developed by Ansh
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        # Display the uploaded image
        st.image(test_image, use_column_width=True, caption="Uploaded Image")
        
        # Predict button
        if st.button("Predict"):
            st.snow()
            st.write("Our Prediction:")
            result_index = model_prediction(test_image)
            
            # Reading Labels
            class_name =['Apple___Apple_scab',
                         'Apple___Black_rot',
                         'Apple___Cedar_apple_rust',
                         'Apple___healthy',
                         'Blueberry___healthy',
                         'Cherry_(including_sour)___Powdery_mildew',
                         'Cherry_(including_sour)___healthy',
                         'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                         'Corn_(maize)___Common_rust_',
                         'Corn_(maize)___Northern_Leaf_Blight',
                         'Corn_(maize)___healthy',
                         'Grape___Black_rot',
                         'Grape___Esca_(Black_Measles)',
                         'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                         'Grape___healthy',
                         'Orange___Haunglongbing_(Citrus_greening)',
                         'Peach___Bacterial_spot',
                         'Peach___healthy',
                         'Pepper,_bell___Bacterial_spot',
                         'Pepper,_bell___healthy',
                         'Potato___Early_blight',
                         'Potato___Late_blight',
                         'Potato___healthy',
                         'Raspberry___healthy',
                         'Soybean___healthy',
                         'Squash___Powdery_mildew',
                         'Strawberry___Leaf_scorch',
                         'Strawberry___healthy',
                         'Tomato___Bacterial_spot',
                         'Tomato___Early_blight',
                         'Tomato___Late_blight',
                         'Tomato___Leaf_Mold',
                         'Tomato___Septoria_leaf_spot',
                         'Tomato___Spider_mites Two-spotted_spider_mite',
                         'Tomato___Target_Spot',
                         'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                         'Tomato___Tomato_mosaic_virus',
                         'Tomato___healthy']
            st.success(f"Model is Predicting it's a {class_name[result_index]}")
