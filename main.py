import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

from util import classify


# set title
st.title('Bone Fracture classification')

st.write("This is a classifier for Bone Fracture")

st.write("Built with PyTorch/Streamlit. Just upload an image from your computer, and it will be classified.")
# set header
st.header('Please upload a Fracture X-ray image')

# upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

if st.button('Submit'):
    # Load the classifier model
    model = load_model('boneClassificationCNN.h5')

    # Load class names
    with open('labels.txt', 'r') as f:
        class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
        f.close()
    # display image
    if file is not None:
        image = Image.open(file).convert('RGB')

        # Resize image to 200x200 pixels
        image = image.resize((200, 200))
        image.thumbnail((50, 50))

        st.image(image, width=400)

        # classify image
        class_name, conf_score = classify(image, model, class_names)

        # write classification
        # check if class name is in list of known labels
        if class_name in class_names:
            # write classification
            st.write("This Person has {} type of Bone Fracture".format(class_name))
            st.write("### score: {}%".format(int(conf_score * 1000) / 10))
        else:
            class_name="Sorry, we don't classify them"
            st.write("## {}".format(class_name))