from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import base64

# uvicorn main:app --reload
# 245,111,66 
# 158,245,66
# 0,0,255    
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("best.pt") 

# --- กำหนดข้อมูลและสีตายตัว (Fixed Colors) ประจำแต่ละคลาส ---
# ระบบสีของ OpenCV คือ BGR (Blue, Green, Red)
CLASS_INFO = {
    0: {"th": "สิ่งเจือปน", "en": "Impurity", "color": (0, 0, 240)},  
    1: {"th": "ข้าวเจ้า", "en": "Rice", "color": (235, 128, 52)},    
    2: {"th": "ข้าวเหนียว", "en": "Sticky-Rice", "color": (50, 168, 90)}     
}

# โค้ดส่วนนี้จะทำหน้าที่เปิดหน้าเว็บเวลาคนพิมพ์ลิงก์เข้ามา
# @app.get("/", response_class=HTMLResponse)
# async def read_index():
#     with open("index.html", "r", encoding="utf-8") as f:
#         return f.read()

@app.post("/predict/")
async def predict_rice(
    file: UploadFile = File(...),
    target_rice: str = Form("ไม่ระบุ")
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8) 
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)
    
    counts = {}
    target_count = 0
    impurity_count = 0

    annotated_img = img.copy()

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # ดึงข้อมูลชื่อและสีที่ฟิกไว้จาก CLASS_INFO
            info = CLASS_INFO.get(cls_id, {"th": "ไม่ทราบ", "en": "Unknown", "color": (255, 0, 0)})
            cls_name_th = info["th"]
            cls_name_en = info["en"]
            box_color = info["color"] # <-- ใช้สีนี้วาดกรอบเสมอ ไม่ว่าจะเลือกโหมดไหน

            # นับจำนวนรวม
            counts[cls_name_th] = counts.get(cls_name_th, 0) + 1

            # --- ลอจิกการคำนวณเป้าหมายเพื่อหาเปอร์เซ็นต์ (อิงตาม Dropdown) ---
            if target_rice == "ไม่ระบุ":
                if cls_name_th in ["ข้าวเจ้า", "ข้าวเหนียว"]:
                    target_count += 1
                else:
                    impurity_count += 1
            else:
                if cls_name_th == target_rice:
                    target_count += 1
                else:
                    impurity_count += 1

            # --- วาดกรอบและข้อความด้วยสีตายตัว ---
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # วาดเส้นกรอบ
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)
            
            # วาดป้ายชื่อ
            label = f"{cls_name_en} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_img, (x1, y1 - 20), (x1 + w, y1), box_color, -1)
            cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    _, buffer = cv2.imencode('.jpg', annotated_img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(content={
        "counts": counts,
        "target_rice": target_rice,
        "target_count": target_count,
        "impurity_count": impurity_count,
        "image": f"data:image/jpeg;base64,{img_base64}"
    })