import streamlit as st
import zipfile
import tempfile
import os
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO

# Load model
model = load_model('mobilenetv2_hyper_tuned_model.h5')
class_labels = ['clean_tackle', 'foul']
label_colors = {'clean_tackle': (0, 255, 0), 'foul': (0, 0, 255)}

# Helper function for image prediction
def predict_image(image):
    resized = image.resize((224, 224))
    image_array = img_to_array(resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array, verbose=0)[0][0]
    label = class_labels[int(prediction > 0.5)]
    return label

# Draw label on image
def draw_label(image, label):
    img_np = np.array(image)
    h, w, _ = img_np.shape
    color = label_colors[label]
    border_thickness = 10

    # Add border
    img_np = cv2.copyMakeBorder(
        img_np,
        border_thickness,
        border_thickness,
        border_thickness,
        border_thickness,
        cv2.BORDER_CONSTANT,
        value=color
    )

    # Text settings
    label_text = label.replace('_', ' ').title()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3

    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

    # Calculate centered position
    x = (img_np.shape[1] - text_width) // 2
    y = border_thickness + text_height + 10  # a little padding below the top border

    # Draw text
    cv2.putText(img_np, label_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))



# Video processing
def process_video(video_bytes):
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(video_bytes)
    temp_input.close()

    cap = cv2.VideoCapture(temp_input.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output.name, fourcc, fps, (width + 20, height + 20))

    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = frame.copy()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, frame_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contact_detected = any(cv2.contourArea(c) > 500 for c in contours)

        label = None
        if contact_detected:
            resized = cv2.resize(frame, (224, 224))
            image_array = img_to_array(resized) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            prediction = model.predict(image_array, verbose=0)[0][0]
            label = class_labels[int(prediction > 0.5)]

        # Add border and label
        border_color = label_colors[label] if label else (0, 0, 0)
        output_frame = cv2.copyMakeBorder(output_frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=border_color)
        if label:
            # Centered text
            label_text = label.replace('_', ' ').title()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3

            (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
            x = (output_frame.shape[1] - text_width) // 2
            y = 40 + text_height

            cv2.putText(output_frame, label_text, (x, y), font, font_scale, border_color, thickness, cv2.LINE_AA)

        out.write(output_frame)
        prev_gray = frame_gray

    cap.release()
    out.release()

    return temp_output.name

# Streamlit UI
st.title("Tackle Classifier: Clean Tackle vs Foul")
uploaded_file = st.file_uploader("Upload image, video or zip", type=["jpg", "jpeg", "png", "mp4", "zip"])

if uploaded_file:
    suffix = uploaded_file.name.split('.')[-1].lower()

    if suffix in ["jpg", "jpeg", "png"]:
        image = Image.open(uploaded_file).convert('RGB')
        label = predict_image(image)
        labeled_img = draw_label(image, label)
        st.image(labeled_img, caption=label.replace('_', ' ').title(), use_container_width=True)
        buffered = BytesIO()
        labeled_img.save(buffered, format="PNG")
        st.download_button("Download Image", buffered.getvalue(), file_name="labeled_image.png")

    elif suffix == "mp4":
        result_path = process_video(uploaded_file.read())
        st.video(result_path)
        with open(result_path, "rb") as video_file:
            video_bytes = video_file.read()
            st.video(video_bytes)
            st.download_button("Download Video", video_bytes, file_name="labeled_video.mp4")

    elif suffix == "zip":
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.read())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)

            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            for root, _, files in os.walk(tmp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.lower().endswith((".jpg", ".jpeg", ".png")):
                        img = Image.open(file_path).convert('RGB')
                        label = predict_image(img)
                        labeled = draw_label(img, label)
                        labeled.save(os.path.join(output_dir, file))

                    elif file.lower().endswith(".mp4"):
                        with open(file_path, "rb") as vf:
                            output_path = process_video(vf.read())
                            os.rename(output_path, os.path.join(output_dir, file))
            

            # Zip the output files
            result_zip = os.path.join(tmp_dir, "labeled_outputs.zip")
            with zipfile.ZipFile(result_zip, 'w') as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        zipf.write(os.path.join(root, file), arcname=file)

            with open(result_zip, "rb") as zf:
                st.download_button("Download Labeled ZIP", zf.read(), file_name="labeled_outputs.zip")
