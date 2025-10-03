import streamlit as st
import cv2
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
import pytz
import gspread
from google.oauth2.service_account import Credentials
from sklearn.neighbors import KNeighborsClassifier
import os
import time

# ================== KONFIGURASI ==================
WIB = pytz.timezone("Asia/Jakarta")
ENCODINGS_PATH = "face_encodings.pkl"
ATTENDANCE_PATH = "attendance.csv"
SHEET_ID = "18onh5sCXMS0KWm40BPUmzCWi3ReayHgJSaWB47GYiFA"
SHEET_NAME = "Absensi"

# Load Haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ================== GOOGLE SHEETS ==================
def connect_gsheet():
    creds = Credentials.from_service_account_file(
        "credentials.json",
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)

def append_to_gsheet(row):
    try:
        ws = connect_gsheet()
        ws.append_row(row)
    except Exception as e:
        st.error(f"❌ Gagal update Google Sheet: {e}")

# ================== DATA WAJAH ==================
def load_known_faces():
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
            return data if isinstance(data, dict) else {"names": [], "embeddings": []}
    except FileNotFoundError:
        return {"names": [], "embeddings": []}

def save_known_faces(data):
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

def log_attendance(name):
    if name == "Unknown":
        return False
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Nama", "Waktu"])
    now = datetime.now(WIB).strftime("%Y-%m-%d %H:%M:%S")
    new_row = pd.DataFrame([[name, now]], columns=["Nama", "Waktu"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(ATTENDANCE_PATH, index=False)
    append_to_gsheet([name, now])  # update ke Google Sheet
    return True

def detect_face_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 3, minSize=(50, 50))
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48))
        hist = cv2.calcHist([roi], [0], None, [48], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist, (x, y, w, h)
    return None, None

# ================== APP ==================
faces_data = load_known_faces()
st.set_page_config(page_title="Absensi Face ID", layout="wide")
st.title("📸 Absensi Face ID (Light Version)")
tab1, tab2, tab3 = st.tabs(["Absensi", "Pendaftaran", "Laporan"])

# -------- ABSENSI --------
with tab1:
    st.header("Absensi dengan Kamera")
    clf = None
    if faces_data["names"] and faces_data["embeddings"]:
        X = np.vstack(faces_data["embeddings"])
        y = np.array(faces_data["names"])
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(X, y)
        st.success(f"✅ Model siap - {len(faces_data['names'])} wajah")
    else:
        st.warning("⚠️ Belum ada data wajah")

    photo = st.camera_input("Ambil foto untuk absensi")
    if photo and clf:
        img = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), 1)
        emb, bbox = detect_face_features(img)
        if emb is not None:
            name = clf.predict([emb])[0]
            x, y, w, h = bbox
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img, name, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            st.image(img, channels="BGR")
            if log_attendance(name):
                st.success(f"✅ Absensi dicatat untuk {name}")
        else:
            st.error("❌ Wajah tidak terdeteksi")

# -------- PENDAFTARAN --------
with tab2:
    st.header("Daftar Wajah Baru")
    new_name = st.text_input("Nama:")
    photo = st.camera_input("Ambil foto wajah")
    if photo and new_name:
        img = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), 1)
        emb, bbox = detect_face_features(img)
        if emb is not None:
            faces_data["names"].append(new_name)
            faces_data["embeddings"].append(emb.tolist())
            save_known_faces(faces_data)
            st.success(f"✅ Wajah {new_name} disimpan")
        else:
            st.error("❌ Wajah tidak terdeteksi")

# -------- LAPORAN --------
with tab3:
    st.header("Laporan Absensi")
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
        st.dataframe(df)
    except FileNotFoundError:
        st.info("Belum ada absensi")
