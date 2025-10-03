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
import json
import threading

# Lock untuk prevent concurrent writes
attendance_lock = threading.Lock()

# ================== KONFIGURASI ==================
st.set_page_config(page_title="Absensi Face ID", layout="wide", initial_sidebar_state="collapsed")

WIB = pytz.timezone("Asia/Jakarta")
ENCODINGS_PATH = "face_encodings.pkl"
ATTENDANCE_PATH = "attendance.csv"
SHEET_ID = "18onh5sCXMS0KWm40BPUmzCWi3ReayHgJSaWB47GYiFA"
SHEET_NAME = "Absensi"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Konfigurasi Funnel dan Leader
FUNNELS = {
    "SF": "Short Form",
    "FF": "Full Form"
}

LEADERS = {
    "Analia": "R Analia Rizki Berliana Sari",
    "Salsabila": "Salsabila Azzahra Maulidina"
}

# ================== GOOGLE SHEETS ==================
@st.cache_resource
def get_gsheet_client():
    """Cache Google Sheets connection"""
    try:
        # Coba dari Streamlit secrets (untuk cloud)
        if "gcp_service_account" in st.secrets:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(
                creds_dict,
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
        # Fallback ke file lokal (untuk development)
        else:
            creds = Credentials.from_service_account_file(
                "credentials.json",
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
        
        client = gspread.authorize(creds)
        return client
    except Exception as e:
        st.warning(f"Google Sheets tidak terhubung: {e}")
        return None

def append_to_gsheet(row):
    """Append data ke Google Sheets"""
    try:
        client = get_gsheet_client()
        if client:
            sheet = client.open_by_key(SHEET_ID).worksheet(SHEET_NAME)
            sheet.append_row(row)
            return True
        return False
    except Exception as e:
        st.error(f"Gagal update Google Sheet: {e}")
        return False

# ================== DATA WAJAH ==================
def load_known_faces():
    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                # Validasi panjang embedding
                if data["embeddings"]:
                    expected_len = len(np.array(data["embeddings"][0]).flatten())
                    valid_names = []
                    valid_embeddings = []
                    for name, emb in zip(data["names"], data["embeddings"]):
                        emb_flat = np.array(emb).flatten()
                        if len(emb_flat) == expected_len:
                            valid_names.append(name)
                            valid_embeddings.append(emb)
                    return {"names": valid_names, "embeddings": valid_embeddings}
                return data
            else:
                return {"names": [], "embeddings": []}
    except FileNotFoundError:
        return {"names": [], "embeddings": []}

def save_known_faces(data):
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)

def log_attendance(name):
    """Catat absensi dengan validasi window 3 menit per jam"""
    if name == "Unknown":
        return False, "Wajah tidak dikenali"
    
    now = datetime.now(WIB)
    current_hour = now.hour
    current_minute = now.minute
    
    # Validasi jam operasional (09:00 - 14:59)
    if current_hour < 9 or current_hour >= 15:
        return False, f"Absensi hanya bisa dilakukan jam 09:00 - 15:00. Sekarang {now.strftime('%H:%M')}"
    
    # Validasi window 3 menit (00-03 setiap jam)
    if current_minute > 3:
        next_hour = current_hour + 1
        if next_hour >= 15:
            return False, f"Window absensi sudah lewat. Jam absensi berikutnya: Besok jam 09:00"
        return False, f"Window absensi sudah lewat. Jam absensi berikutnya: {next_hour:02d}:00-{next_hour:02d}:03"
    
    # Gunakan lock untuk prevent concurrent writes
    with attendance_lock:
        try:
            df = pd.read_csv(ATTENDANCE_PATH)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Nama", "Waktu"])
        
        today = now.strftime("%Y-%m-%d")
        current_hour_slot = now.strftime("%Y-%m-%d %H")  # Format: 2025-10-03 09
        
        # Cek apakah sudah absen di jam yang sama hari ini
        if not df.empty:
            user_entries = df[df["Nama"] == name]
            if not user_entries.empty:
                for _, row in user_entries.iterrows():
                    entry_time_str = row["Waktu"]
                    # Cek apakah ada absen di jam slot yang sama
                    entry_hour_slot = entry_time_str[:13]  # Ambil YYYY-MM-DD HH
                    if entry_hour_slot == current_hour_slot:
                        return False, f"{name} sudah absen pada jam {now.strftime('%H')}:00 hari ini"
        
        # Simpan ke CSV
        now_str = now.strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame([[name, now_str]], columns=["Nama", "Waktu"])
        df = pd.concat([df, new_row], ignore_index=True)
        
        try:
            df.to_csv(ATTENDANCE_PATH, index=False)
        except Exception as e:
            return False, f"Error menyimpan: {e}"
    
    # Update ke Google Sheets
    gsheet_success = append_to_gsheet([name, now_str])
    
    if gsheet_success:
        return True, f"Absensi berhasil pada {now_str}"
    else:
        return True, f"Absensi berhasil pada {now_str}"

def detect_face_features(img):
    """Deteksi wajah dan extract features dengan LBP (Local Binary Patterns)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
    faces = face_cascade.detectMultiScale(small_gray, 1.2, 5, minSize=(30, 30))
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        x, y, w, h = x*2, y*2, w*2, h*2
        
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (100, 100))
        
        # Equalize histogram untuk konsistensi pencahayaan
        roi = cv2.equalizeHist(roi)
        
        # Extract multiple features untuk akurasi lebih baik
        # 1. Histogram gabungan dari multiple regions
        h_third = roi.shape[0] // 3
        w_third = roi.shape[1] // 3
        
        features = []
        
        # Divide face into 9 regions (3x3 grid) dan hitung histogram tiap region
        for i in range(3):
            for j in range(3):
                region = roi[i*h_third:(i+1)*h_third, j*w_third:(j+1)*w_third]
                hist = cv2.calcHist([region], [0], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                features.extend(hist)
        
        # 2. Tambahkan edge detection features
        edges = cv2.Canny(roi, 50, 150)
        edge_hist = cv2.calcHist([edges], [0], None, [16], [0, 256])
        edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
        features.extend(edge_hist)
        
        # 3. Tambahkan geometric features (posisi dan ukuran relatif)
        img_h, img_w = img.shape[:2]
        features.extend([
            x / img_w,
            y / img_h,
            w / img_w,
            h / img_h,
            (w * h) / (img_w * img_h)  # area ratio
        ])
        
        return np.array(features, dtype=np.float32), (x, y, w, h)
    
    return None, None

# ================== LOAD DATA ==================
faces_data = load_known_faces()

# ================== UI ==================
st.title("📸 Absensi Face ID")
st.caption(f"🕒 {datetime.now(WIB).strftime('%A, %d %B %Y - %H:%M:%S WIB')}")

tab1, tab2, tab3, tab4 = st.tabs(["📷 Absensi", "📝 Pendaftaran", "📊 Laporan", "⚙️ Pengaturan"])

# ============ TAB ABSENSI ============
with tab1:
    st.header("Absensi dengan Kamera")
    
    # Tampilkan waktu dan status
    now = datetime.now(WIB)
    current_hour = now.hour
    current_minute = now.minute
    
    col_time, col_status = st.columns([2, 1])
    
    with col_time:
        st.info(f"🕒 {now.strftime('%H:%M:%S WIB')}")
    
    with col_status:
        # Cek apakah dalam window absensi
        in_window = (9 <= current_hour < 15) and (current_minute <= 3)
        if in_window:
            st.success("✅ Window Absensi")
        elif current_hour < 9 or current_hour >= 15:
            st.error("❌ Di Luar Jam")
        else:
            next_hour = current_hour + 1
            if next_hour < 15:
                st.warning(f"⏳ Window berikutnya: {next_hour:02d}:00")
            else:
                st.warning("⏳ Window berikutnya: Besok")
    
    # Info jadwal absensi
    st.caption("⏰ Window absensi: **09:00-09:03, 10:00-10:03, 11:00-11:03, 12:00-12:03, 13:00-13:03, 14:00-14:03**")
    st.caption("ℹ️ Anda dapat absen 1x per jam di window waktu tersebut")
    
    # Load model
    clf = None
    if faces_data["names"] and faces_data["embeddings"]:
        try:
            X = np.vstack([np.array(e).flatten() for e in faces_data["embeddings"]])
            y = np.array(faces_data["names"])
            
            # Gunakan KNN dengan parameter yang lebih strict
            # n_neighbors=3 untuk lebih robust, metric='euclidean' untuk konsistensi
            clf = KNeighborsClassifier(
                n_neighbors=min(3, len(faces_data["names"])),  # Max 3 atau jumlah data
                weights='distance',  # Weight by distance
                algorithm='ball_tree',
                metric='euclidean'
            )
            clf.fit(X, y)
            st.success(f"Model siap - {len(faces_data['names'])} wajah terdaftar")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.warning("Belum ada data wajah. Silakan daftar di tab Pendaftaran")

    photo = st.camera_input("Ambil foto untuk absensi")
    
    if photo and clf:
        img = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        
        with st.spinner("Mengenali wajah..."):
            emb, bbox = detect_face_features(img)
        
        if emb is not None:
            # Validasi panjang feature vector
            expected_len = len(np.array(faces_data["embeddings"][0]).flatten()) if faces_data["embeddings"] else len(emb)
            if len(emb) != expected_len:
                st.error(f"Feature mismatch! Data lama perlu didaftar ulang. Silakan reset data di tab Pengaturan")
                st.stop()
            
            # Predict dengan probability
            proba = clf.predict_proba([emb])[0]
            max_proba = max(proba)
            name = clf.predict([emb])[0]
            
            # Threshold confidence: minimal 60% yakin
            if max_proba < 0.6:
                name = "Unknown"
                st.warning(f"Confidence terlalu rendah ({max_proba*100:.1f}%). Wajah tidak dikenali dengan pasti")
            
            x, y, w, h = bbox
            
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            
            # Tampilkan nama + confidence
            label = f"{name} ({max_proba*100:.0f}%)"
            cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            st.image(img, channels="BGR", use_container_width=True)
            
            if name != "Unknown":
                success, message = log_attendance(name)
                if success:
                    st.success(message)
                    st.balloons()
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.warning(message)
            else:
                st.error("Wajah tidak dikenali atau confidence rendah. Silakan coba lagi atau daftar wajah baru")
        else:
            st.error("Wajah tidak terdeteksi. Pastikan pencahayaan cukup dan wajah terlihat jelas")

# ============ TAB PENDAFTARAN ============
with tab2:
    st.header("Daftar Wajah Baru")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        new_name = st.text_input("Nama Lengkap:")
    with col2:
        st.metric("Total Terdaftar", len(faces_data["names"]))
    
    if new_name:
        photo = st.camera_input("Ambil foto wajah untuk pendaftaran")
        
        if photo:
            img = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            
            with st.spinner("Memproses wajah..."):
                emb, bbox = detect_face_features(img)
            
            if emb is not None:
                x, y, w, h = bbox
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(img, new_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                st.image(img, channels="BGR", use_container_width=True)
                st.info("Pastikan wajah Anda terlihat jelas dan pencahayaan cukup untuk hasil terbaik")
                
                if st.button("💾 Simpan Wajah", type="primary"):
                    # Cek duplikasi nama
                    if new_name in faces_data["names"]:
                        st.error(f"Nama '{new_name}' sudah terdaftar. Gunakan nama lain")
                    else:
                        faces_data["names"].append(new_name)
                        faces_data["embeddings"].append(emb.tolist())
                        save_known_faces(faces_data)
                        st.success(f"Wajah {new_name} berhasil disimpan!")
                        st.balloons()
                        time.sleep(1.5)
                        st.rerun()
            else:
                st.error("Wajah tidak terdeteksi")
    else:
        st.info("Masukkan nama terlebih dahulu")
    
    # Daftar terdaftar
    if faces_data["names"]:
        with st.expander("Lihat Daftar Terdaftar"):
            for i, name in enumerate(faces_data["names"], 1):
                st.write(f"{i}. {name}")

# ============ TAB LAPORAN ============
with tab3:
    st.header("Laporan Absensi")
    
    try:
        df = pd.read_csv(ATTENDANCE_PATH)
        today = datetime.now(WIB).strftime("%Y-%m-%d")
        today_df = df[df["Waktu"].str.contains(today)]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Absensi Hari Ini", len(today_df))
        with col2:
            st.metric("Total Terdaftar", len(faces_data["names"]))
        with col3:
            unique_today = today_df["Nama"].nunique() if not today_df.empty else 0
            st.metric("Orang Unik Hari Ini", unique_today)
        
        st.subheader("Riwayat Absensi Hari Ini")
        if not today_df.empty:
            # Tambahkan kolom jam untuk analisis
            today_df_copy = today_df.copy()
            today_df_copy["Jam"] = pd.to_datetime(today_df_copy["Waktu"]).dt.strftime("%H:%M")
            
            st.dataframe(
                today_df_copy[["Nama", "Jam", "Waktu"]].sort_values(by="Waktu", ascending=False),
                use_container_width=True,
                hide_index=True
            )
            
            # Summary per orang
            with st.expander("📊 Summary per Orang"):
                summary = today_df_copy.groupby("Nama").agg({
                    "Waktu": "count"
                }).rename(columns={"Waktu": "Total Absensi"})
                st.dataframe(summary, use_container_width=True)
        else:
            st.info("Belum ada yang absen hari ini")
        
        with st.expander("📜 Semua Riwayat"):
            st.dataframe(df.sort_values(by="Waktu", ascending=False), use_container_width=True, hide_index=True)
            
    except FileNotFoundError:
        st.info("Belum ada data absensi")

# ============ TAB PENGATURAN ============
with tab4:
    st.header("Pengaturan")
    
    # Google Sheets status
    st.subheader("Status Google Sheets")
    client = get_gsheet_client()
    if client:
        st.success("Terhubung dengan Google Sheets")
    else:
        st.warning("Tidak terhubung - Data hanya tersimpan di CSV")
        with st.expander("Cara Setup Google Sheets"):
            st.markdown("""
            1. Buat Service Account di Google Cloud Console
            2. Download credentials.json
            3. Di Streamlit Cloud: Settings > Secrets
            4. Tambahkan:
            ```toml
            [gcp_service_account]
            type = "service_account"
            project_id = "your-project-id"
            private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
            client_email = "your-email@project.iam.gserviceaccount.com"
            ```
            """)
    
    st.divider()
    
    # Reset data
    st.subheader("Reset Data")
    
    st.warning("⚠️ **PENTING**: Jika update sistem face recognition, semua data wajah harus didaftar ulang!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Refresh Halaman"):
            st.rerun()
    
    with col2:
        if st.button("🗑️ Hapus Semua Data", type="secondary"):
            if os.path.exists(ENCODINGS_PATH):
                os.remove(ENCODINGS_PATH)
            if os.path.exists(ATTENDANCE_PATH):
                os.remove(ATTENDANCE_PATH)
            st.success("Data berhasil dihapus!")
            st.info("Silakan daftar ulang semua wajah di tab Pendaftaran")
            time.sleep(1)
            st.rerun()
