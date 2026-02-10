import torch
import json
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import pytesseract
import os

# ==========================================
# 1. CONFIGURATION (CHANGE THIS PATH!)

IMAGE_PATH = "/Users/sumitranjan/Downloads/my_receipt.jpg" 
MODEL_PATH = "./my_final_model"

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def normalize_box(box, width, height):
    return [
        max(0, min(1000, int(1000 * (box[0] / width)))),
        max(0, min(1000, int(1000 * (box[1] / height)))),
        max(0, min(1000, int(1000 * (box[2] / width)))),
        max(0, min(1000, int(1000 * (box[3] / height))))
    ]

def get_ocr_data(image):
    """
    Runs Tesseract OCR on the image to get Words and Boxes.
    """
    print("👀 Running OCR (Tesseract)...")
    try:
        # Pytesseract returns a dataframe with columns: left, top, width, height, text...
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError:
        print("\n❌ CRITICAL ERROR: Tesseract is not installed on your system.")
        print("   Please run: 'brew install tesseract' (Mac) or 'sudo apt install tesseract-ocr' (Linux)")
        exit()

    words = []
    boxes = []
    width, height = image.size

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        # Skip empty text (noise)
        txt = data['text'][i].strip()
        if not txt:
            continue

        # Get coordinates
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        
        # Convert to [x1, y1, x2, y2]
        box = [x, y, x + w, y + h]
        
        words.append(txt)
        boxes.append(normalize_box(box, width, height))
        
    return words, boxes

# ==========================================
# 3. MAIN INFERENCE PIPELINE
# ==========================================

# A. Load Image
if not os.path.exists(IMAGE_PATH):
    print(f"❌ Error: File not found at {IMAGE_PATH}")
    exit()

print(f"📂 Loading Image: {IMAGE_PATH}")
image = Image.open(IMAGE_PATH).convert("RGB")

# B. Load Model
print(f"🤖 Loading Model from {MODEL_PATH}...")
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH)
processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH, apply_ocr=False)

# C. Get Inputs (The "Eyes")
words, boxes = get_ocr_data(image)
print(f"   Found {len(words)} words in the image.")

if len(words) == 0:
    print("❌ Error: OCR found no text. Is the image clear?")
    exit()

# D. Run Prediction (The "Brain")
print("🧠 Analyzing Layout...")
inputs = processor(
    image, 
    words, 
    boxes=boxes, 
    return_tensors="pt"
)

with torch.no_grad():
    outputs = model(**inputs)

# E. Reconstruct JSON
predictions = outputs.logits.argmax(-1).squeeze().tolist()
labels = [model.config.id2label[p] for p in predictions]

final_json = {"menu": [], "total": {}}
current_item = {}

print("📝 Constructing Receipt JSON...")
for word, label in zip(words, labels):
    if label == "O": continue
    
    parts = label.split("-")
    if len(parts) < 2: continue
    category = parts[1]
    
    if "menu" in category:
        field = category.split(".")[1]
        
        # Logic: If we see a new Name (B-menu.nm), save the previous item
        if field == "nm" and label.startswith("B-") and "nm" in current_item:
            final_json["menu"].append(current_item)
            current_item = {}
        
        if field in current_item:
            current_item[field] += " " + word
        else:
            current_item[field] = word
            
    elif "total" in category:
        field = category.split(".")[1]
        if field in final_json["total"]:
            final_json["total"][field] += " " + word
        else:
            final_json["total"][field] = word

# Don't forget the last item
if current_item:
    final_json["menu"].append(current_item)

# ==========================================
# 4. OUTPUT
# ==========================================
print("\n" + "="*40)
print("✅  FINAL RESULT (Local File)")
print("="*40)
print(json.dumps(final_json, indent=2))