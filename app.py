import torch._dynamo.utils  # Cached first to resolve Python 3.11/3.12 loading bugs

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import streamlit as st
import json
import io
import zipfile
from PIL import Image
st.set_page_config(page_title="MVTec Hazelnut Quality Inspector", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .main {
        background-color: #0f172a;
    }
    div[data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #10b981;
    }
    .stButton>button {
        background-color: #10b981 !important;
        color: #0f172a !important;
        font-weight: bold;
        border-radius: 6px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)



class ResNetFusionMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(weights=None) # Initialized via weights file loading
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Feature Fusion Segmentation Decoder (FPN style)
        self.lat4 = nn.Conv2d(512, 128, kernel_size=1)
        self.lat3 = nn.Conv2d(256, 128, kernel_size=1)
        self.lat2 = nn.Conv2d(128, 128, kernel_size=1)
        self.lat1 = nn.Conv2d(64, 128, kernel_size=1)
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_final = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.smooth1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.smooth2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.seg_head = nn.Conv2d(32, 2, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.clf_head = nn.Linear(512, 2)
        
    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.layer1(x0) 
        x2 = self.layer2(x1) 
        x3 = self.layer3(x2) 
        x4 = self.layer4(x3) 
        
        clf_in = self.avgpool(x4)
        clf_in = torch.flatten(clf_in, 1)
        clf_out = self.clf_head(clf_in)
        
        p4 = self.lat4(x4)                      
        p3 = self.lat3(x3) + self.up(p4)        
        p2 = self.lat2(x2) + self.up(p3)        
        p1 = self.lat1(x1) + self.up(p2)        
        
        d1 = self.smooth1(p1)                   
        d2 = self.up_final(d1)                  
        d3 = self.smooth2(d2)                   
        seg_out = self.seg_head(d3)             
        
        return clf_out, seg_out




class CustomConvNetMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2) 
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2) 
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2) 
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2) 
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2) 
        )
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) 
        self.dec1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) 
        self.dec2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
        self.dec3 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=4) 
        self.dec4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.seg_head = nn.Conv2d(32, 2, kernel_size=1)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.clf_head = nn.Linear(512, 2)
        
    def forward(self, x):
        x1 = self.enc1(x)  
        x2 = self.enc2(x1) 
        x3 = self.enc3(x2) 
        x4 = self.enc4(x3) 
        x5 = self.enc5(x4) 
        
        up1_out = self.up1(x5)
        merge1 = torch.cat([up1_out, x4], dim=1)
        d1 = self.dec1(merge1)
        
        up2_out = self.up2(d1)
        merge2 = torch.cat([up2_out, x3], dim=1)
        d2 = self.dec2(merge2)
        
        up3_out = self.up3(d2)
        merge3 = torch.cat([up3_out, x2], dim=1)
        d3 = self.dec3(merge3)
        
        up4_out = self.up4(d3)
        d4 = self.dec4(up4_out)
        seg_out = self.seg_head(d4)
        
        clf_in = self.avgpool(x5)
        clf_in = torch.flatten(clf_in, 1)
        clf_out = self.clf_head(clf_in)
        
        return clf_out, seg_out



st.sidebar.title("Quality Inspection Console")
st.sidebar.markdown("### Agricultural Defect Analyzer")

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

if st.sidebar.button(" Clear & Reset Console"):
    st.session_state["uploader_key"] += 1
    st.rerun()

# 1. Model Selector Radio Option
selected_model_option = st.sidebar.radio(
    "Choose Active Architecture",
    ("Pre-trained ResNet Feature Fusion (FPN)", "Custom ConvNet From-Scratch")
)

# 2. Probability class threshold
prob_threshold = st.sidebar.slider(
    "Defect Segmentation Sensitivity", 
    min_value=0.10, 
    max_value=1.00, 
    value=0.50, 
    step=0.05
)

@st.cache_resource
def load_selected_model(model_name):
    # Selects device type dynamically (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device_type = 'cuda'
    elif torch.backends.mps.is_available():
        device_type = 'mps'
    else:
        device_type = 'cpu'
    
    if model_name == "Pre-trained ResNet Feature Fusion (FPN)":
        model = ResNetFusionMultiTask()
        weights_path = "mvtec_multitask_hazelnut_fusion.pth"
    else:
        model = CustomConvNetMultiTask()
        weights_path = "mvtec_multitask_hazelnut_convnet.pth"
        
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device_type))
        st.sidebar.info(f"Loaded {model_name} weights successfully.")
    else:
        st.sidebar.warning(f"⚠️ Checkpoint file '{weights_path}' was not found. Defaults to random initializations.")
        
    model.to(device_type)
    model.eval()
    return model, device_type

active_model, active_device = load_selected_model(selected_model_option)
st.sidebar.success(f"Execution Device: {active_device.upper()}")


# ==========================================
# STREAMLIT MAIN INTERFACE
# ==========================================
st.title("🌰 Automated Hazelnut Quality Inspection Suite")
st.write("A deep learning framework performing real-time classification, pixel-level defect segmentation, and anomaly object bounding-box detection.")

uploaded_files = st.file_uploader(
    "Upload Hazelnut Scan Images (.jpg, .jpeg, .png)", 
    accept_multiple_files=True,
    key=f"file_uploader_{st.session_state['uploader_key']}"
)

class_names = ["Normal (Good)", "Anomalous (Defect)"]

if uploaded_files:
    st.markdown("##  Unified Multi-Task Inference Results")
    
    metadata_json_results = {}
    processed_images_for_zip = []
    
    for idx, file in enumerate(uploaded_files):
        # Decode uploaded image BGR -> RGB
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        
        st.subheader(f"Inference: {file.name}")
        
        # Preprocessing matching dataset normalizations
        img_resized = cv2.resize(img_rgb, (224, 224))
        img_norm = (img_resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor_input = torch.tensor(img_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(active_device)
        
        with torch.no_grad():
            clf_out, seg_out = active_model(tensor_input)
            
            # Classification inference
            probs = torch.softmax(clf_out.squeeze(0), dim=0).cpu().numpy()
            pred_class_idx = np.argmax(probs)
            pred_class_name = class_names[pred_class_idx]
            confidence_score = probs[pred_class_idx]
            
            # Segmentation inference (softmax/argmax over channels)
            seg_probs = torch.softmax(seg_out, dim=1).squeeze(0).cpu().numpy()
            # Probability map of defect class (index 1)
            pred_seg_prob = seg_probs[1]
            
        # Re-scale target predictions back to original dimensions
        pred_seg_prob_resized = cv2.resize(pred_seg_prob, (w, h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (pred_seg_prob_resized > prob_threshold).astype(np.uint8) * 255
        
        # Calculate localized boundaries for Bounding Box (Object Detection)
        y_indices, x_indices = np.where(binary_mask > 0)
        has_detection = len(y_indices) > 0
        
        bbox_coords = []
        if has_detection:
            ymin, ymax = int(y_indices.min()), int(y_indices.max())
            xmin, xmax = int(x_indices.min()), int(x_indices.max())
            bbox_coords = [xmin, ymin, xmax, ymax]
            
        # 1. Generate Segmentation Overlay image
        overlay_img = img_rgb.copy()
        color_mask = np.zeros_like(img_rgb)
        color_mask[binary_mask > 0] = [239, 68, 68]  # Red overlay for segmentation
        cv2.addWeighted(color_mask, 0.4, overlay_img, 0.6, 0, overlay_img)
        
        # 2. Generate Detection Bounding Box image
        detect_img = img_rgb.copy()
        if has_detection:
            # Draw orange bounding box around predicted defect coordinates
            cv2.rectangle(detect_img, (xmin, ymin), (xmax, ymax), (249, 115, 22), 4)
            cv2.putText(
                detect_img, "DEFECT", (xmin, max(ymin - 10, 15)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (249, 115, 22), 2, cv2.LINE_AA
            )
            
        # Compile metadata for downstream export
        metadata_json_results[file.name] = {
            "prediction_class": pred_class_name,
            "prediction_confidence": f"{confidence_score * 100:.2f}%",
            "is_defect_segmented": bool(has_detection),
            "dynamic_bounding_box_coordinates": bbox_coords,
            "segmented_pixels_area": int(np.sum(binary_mask > 0))
        }
        
        # Encode overlay display image to BGR bytes for download
        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
        img_bytes = img_encoded.tobytes()
        processed_images_for_zip.append((f"segmented_{file.name}", img_bytes))
        
        # Render Multi-Task Columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img_rgb, caption="Raw Hazelnut Input", use_container_width=True)
            # Display classification metrics directly below the input
            if pred_class_idx == 0:
                st.markdown(f"**Classification Prediction:** :green[{pred_class_name}] ({confidence_score * 100:.2f}%)")
            else:
                st.markdown(f"**Classification Prediction:** :red[{pred_class_name}] ({confidence_score * 100:.2f}%)")
                
        with col2:
            st.image(overlay_img, caption="FPN Anomaly Mask Overlay", use_container_width=True)
            st.download_button(
                label=f"⬇️ Download Mask Overlay",
                data=img_bytes,
                file_name=f"overlay_{file.name}",
                mime="image/jpeg",
                key=f"download_mask_btn_{idx}"
            )
            
        with col3:
            st.image(detect_img, caption="Dynamic Defect Bounding Box", use_container_width=True)
            # Encode detection display image to BGR bytes for download
            _, detect_encoded = cv2.imencode('.jpg', cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR))
            detect_bytes = detect_encoded.tobytes()
            st.download_button(
                label=f"⬇️ Download Bounding Box",
                data=detect_bytes,
                file_name=f"detection_{file.name}",
                mime="image/jpeg",
                key=f"download_detect_btn_{idx}"
            )
            
        st.markdown("---")
        
    # --- Downstream Export Actions ---
    st.markdown("## 📥 Export System Metrics")
    col_batch1, col_batch2 = st.columns(2)
    
    with col_batch1:
        st.markdown("### 🖼️ Download All Mask Overlays")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_name, img_data in processed_images_for_zip:
                zip_file.writestr(file_name, img_data)
        
        st.download_button(
            label="📦 Download All Segmentations (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="hazelnut_quality_segmented_images.zip",
            mime="application/zip"
        )
        
    with col_batch2:
        st.markdown("### 📊 Download System Metadata")
        json_metadata_string = json.dumps(metadata_json_results, indent=4)
        st.download_button(
            label="💾 Download Metrics (JSON)",
            data=json_metadata_string,
            file_name="hazelnut_segmentation_results.json",
            mime="application/json"
        )
        
    with st.expander("Preview Export JSON Metadata Records"):
        st.json(metadata_json_results)