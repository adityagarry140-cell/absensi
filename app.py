import streamlit as st
import streamlit.components.v1 as components
import os
import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import pytz
from sklearn.neighbors import KNeighborsClassifier
import base64

# Pengaturan
st.set_page_config(
    page_title="Absensi Face ID", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

WIB = pytz.timezone('Asia/Jakarta')
ENCODINGS_PATH = 'face_encodings.pkl'
ATTENDANCE_PATH = 'attendance.csv'
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi helper
def load_known_faces():
    try:
        with open(ENCODINGS_PATH, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                names = data.get("names", [])
                embeddings = data.get("embeddings", [])
            else:
                encodings, names = data
                embeddings = encodings
            clean_names, clean_embs = [], []
            target_len = None
            for n, e in zip(names, embeddings):
                e = np.array(e, dtype=np.float32).flatten()
                if target_len is None:
                    target_len = len(e)
                if len(e) == target_len:
                    clean_names.append(n)
                    clean_embs.append(e)
            return {"names": clean_names, "embeddings": clean_embs}
    except FileNotFoundError:
        return {"names": [], "embeddings": []}

def save_known_faces(data):
    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump(data, f)

def log_attendance(name):
    if name == "Unknown":
        return False
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Nama', 'Waktu'])
    if not df.empty:
        last_entry = df[df['Nama'] == name]
        if not last_entry.empty:
            try:
                last_time_str = last_entry['Waktu'].iloc[-1]
                last_time = WIB.localize(datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S'))
                if (datetime.now(WIB) - last_time).total_seconds() < 60:
                    return False
            except:
                pass
    now = datetime.now(WIB).strftime('%Y-%m-%d %H:%M:%S')
    df = pd.concat([df, pd.DataFrame([[name, now]], columns=['Nama', 'Waktu'])], ignore_index=True)
    df.to_csv(ATTENDANCE_PATH, index=False)
    return True

def detect_face_optimized(img):
    h, w = img.shape[:2]
    scale = 0.4
    small = cv2.resize(img, (int(w*scale), int(h*scale)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 3, minSize=(20, 20))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        x, y, w, h = int(x/scale), int(y/scale), int(w/scale), int(h/scale)
        face_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_roi = face_gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        hist = cv2.calcHist([face_roi], [0], None, [48], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        img_h, img_w = img.shape[:2]
        features = np.concatenate([hist, [x/img_w, y/img_h, w/img_w, h/img_h]])
        return features.astype(np.float32), (x, y, w, h)
    return None, None

def decode_base64_image(base64_str):
    img_data = base64.b64decode(base64_str.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# Custom component untuk kamera lightweight
def camera_component(key="camera", width=640, height=480, fps=15):
    html_code = f"""
    <div style="text-align: center;">
        <video id="video-{key}" width="{width}" height="{height}" autoplay playsinline style="max-width: 100%; border: 2px solid #4CAF50; border-radius: 8px;"></video>
        <canvas id="canvas-{key}" width="{width}" height="{height}" style="display:none;"></canvas>
        <br><br>
        <button onclick="captureFrame()" style="padding: 12px 24px; font-size: 16px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;">
            Capture Foto
        </button>
        <div id="status-{key}" style="margin-top: 10px; font-weight: bold;"></div>
    </div>
    
    <script>
    const video = document.getElementById('video-{key}');
    const canvas = document.getElementById('canvas-{key}');
    const status = document.getElementById('status-{key}');
    const ctx = canvas.getContext('2d');
    
    // Request camera dengan constraint ringan
    navigator.mediaDevices.getUserMedia({{
        video: {{
            width: {{ ideal: {width} }},
            height: {{ ideal: {height} }},
            frameRate: {{ ideal: {fps}, max: {fps} }},
            facingMode: "user"
        }}
    }})
    .then(stream => {{
        video.srcObject = stream;
        status.textContent = "Kamera siap";
        status.style.color = "green";
    }})
    .catch(err => {{
        status.textContent = "Error: " + err.message;
        status.style.color = "red";
    }});
    
    function captureFrame() {{
        ctx.drawImage(video, 0, 0, {width}, {height});
        const imageData = canvas.toDataURL('image/jpeg', 0.8);
        
        // Send ke Streamlit
        window.parent.postMessage({{
            type: 'streamlit:setComponentValue',
            value: imageData
        }}, '*');
        
        status.textContent = "Foto diambil, memproses...";
        status.style.color = "blue";
    }}
    </script>
    """
    
    return components.html(html_code, height=height + 100)

# Load data
faces_data = load_known_faces()

# Header
st.title("Sistem Absensi Face ID - Mobile Optimized")

# Tabs
tab1, tab2, tab3 = st.tabs(["Absensi", "Pendaftaran", "Laporan"])

# TAB ABSENSI
with tab1:
    st.header("Absensi Real-time")
    
    # Load model
    clf = None
    if faces_data["names"] and faces_data["embeddings"]:
        try:
            X = np.vstack([np.array(e, dtype=np.float32).flatten() for e in faces_data["embeddings"]])
            y = np.array(faces_data["names"])
            clf = KNeighborsClassifier(n_neighbors=1, algorithm='ball_tree')
            clf.fit(X, y)
            st.success(f"Siap - {len(faces_data['names'])} wajah terdaftar")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Belum ada wajah terdaftar")
    
    st.write(f"Waktu: **{datetime.now(WIB).strftime('%H:%M:%S WIB')}**")
    
    # Camera component dengan resolusi rendah untuk mobile
    captured_image = camera_component(key="absensi", width=480, height=360, fps=15)
    
    if captured_image:
        try:
            img = decode_base64_image(captured_image)
            emb, bbox = detect_face_optimized(img)
            
            if emb is not None and clf:
                name = clf.predict(emb.reshape(1, -1))[0]
                x, y, w, h = bbox
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
                cv2.putText(img, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                
                st.image(img, channels="BGR", use_column_width=True)
                
                if name != "Unknown":
                    if log_attendance(name):
                        st.success(f"Absensi berhasil: {name}")
                        st.balloons()
                    else:
                        st.info(f"{name} sudah absen dalam 1 menit terakhir")
                else:
                    st.error("Wajah tidak dikenali")
            else:
                st.warning("Tidak ada wajah terdeteksi")
        except Exception as e:
            st.error(f"Error processing: {e}")

# TAB PENDAFTARAN
with tab2:
    st.header("Pendaftaran Wajah Baru")
    
    new_name = st.text_input("Nama Lengkap:")
    
    if new_name:
        captured_reg = camera_component(key="register", width=480, height=360, fps=15)
        
        if captured_reg:
            try:
                img = decode_base64_image(captured_reg)
                emb, bbox = detect_face_optimized(img)
                
                if emb is not None:
                    x, y, w, h = bbox
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                    st.image(img, channels="BGR", use_column_width=True)
                    
                    if st.button("Simpan Wajah Ini"):
                        faces_data["names"].append(new_name)
                        faces_data["embeddings"].append(emb.tolist())
                        save_known_faces(faces_data)
                        st.success(f"Wajah {new_name} berhasil disimpan!")
                        st.rerun()
                else:
                    st.error("Wajah tidak terdeteksi")
            except Exception as e:
                st.error(f"Error: {e}")

# TAB LAPORAN
with tab3:
    st.header("Laporan Kehadiran")
    
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
        today = datetime.now(WIB).strftime('%Y-%m-%d')
        today_df = df[df['Waktu'].str.contains(today)]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Hadir Hari Ini", len(today_df) if not today_df.empty else 0)
        with col2:
            st.metric("Total Terdaftar", len(faces_data['names']))
        
        if not today_df.empty:
            st.subheader("Kehadiran Hari Ini")
            st.dataframe(today_df.sort_values(by='Waktu', ascending=False), use_column_width=True)
        else:
            st.info("Belum ada yang absen hari ini")
        
        with st.expander("Semua Riwayat"):
            st.dataframe(df.sort_values(by='Waktu', ascending=False), use_column_width=True)
            
    except FileNotFoundError:
        st.info("Belum ada data kehadiran")
    
    if st.button("Reset Semua Data"):
        if os.path.exists(ENCODINGS_PATH):
            os.remove(ENCODINGS_PATH)
        if os.path.exists(ATTENDANCE_PATH):
            os.remove(ATTENDANCE_PATH)
        st.success("Data berhasil dihapus!")
        st.rerun()