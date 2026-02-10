import json
import numpy as np
from datasets import load_dataset
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, TrainingArguments, Trainer
from transformers.data.data_collator import default_data_collator
import evaluate
import torch
from PIL import Image

# --- 1. CONFIGURATION & LABELS ---
LABELS = [
    "O", 
    "B-menu.nm", "I-menu.nm", 
    "B-menu.cnt", "I-menu.cnt", 
    "B-menu.price", "I-menu.price", 
    "B-menu.unitprice", "I-menu.unitprice", 
    "B-menu.discountprice", "I-menu.discountprice", 
    "B-menu.sub_nm", "I-menu.sub_nm", 
    "B-menu.sub_cnt", "I-menu.sub_cnt", 
    "B-menu.sub_price", "I-menu.sub_price",
    "B-total.total_price", "I-total.total_price", 
    "B-total.cashprice", "I-total.cashprice", 
    "B-total.changeprice", "I-total.changeprice", 
    "B-total.creditcardprice", "I-total.creditcardprice",
    "B-total.emoneyprice", "I-total.emoneyprice", 
    "B-total.menutype_cnt", "I-total.menutype_cnt", 
    "B-total.menuqty_cnt", "I-total.menuqty_cnt",
    "B-sub_total.subtotal_price", "I-sub_total.subtotal_price",
    "B-sub_total.discount_price", "I-sub_total.discount_price",
    "B-sub_total.service_price", "I-sub_total.service_price",
    "B-sub_total.tax_price", "I-sub_total.tax_price",
    "B-void_menu.nm", "I-void_menu.nm",
    "B-void_menu.price", "I-void_menu.price" 
]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for i, label in enumerate(LABELS)}

# --- 2. DATA PROCESSING ---

# Fix 1: Robust Normalization with Clamping
def normalize_box(box, width, height):
    return [
        max(0, min(1000, int(1000 * (box[0] / width)))),
        max(0, min(1000, int(1000 * (box[1] / height)))),
        max(0, min(1000, int(1000 * (box[2] / width)))),
        max(0, min(1000, int(1000 * (box[3] / height))))
    ]

# Fix 2: The Filter is BACK to catch that OSError image
def filter_corrupt_images(example):
    try:
        # Force strict loading to catch errors early
        example['image'].convert("RGB")
        return True
    except Exception:
        return False

def parse_cord_json(examples):
    images = examples['image']
    ground_truths = examples['ground_truth']
    
    all_words = []
    all_boxes = []
    all_labels = []
    
    for img, gt_str in zip(images, ground_truths):
        gt = json.loads(gt_str)
        width, height = img.size
        
        words = []
        boxes = []
        ner_tags = []
        
        if "valid_line" in gt:
            for line in gt["valid_line"]:
                category = line["category"]
                for i, word_info in enumerate(line["words"]):
                    word_text = word_info["text"]
                    
                    q = word_info["quad"]
                    x_coords = [q["x1"], q["x2"], q["x3"], q["x4"]]
                    y_coords = [q["y1"], q["y2"], q["y3"], q["y4"]]
                    box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    
                    prefix = "B-" if i == 0 else "I-"
                    tag_name = f"{prefix}{category}"
                    tag_id = label2id.get(tag_name, label2id["O"])
                    
                    words.append(word_text)
                    boxes.append(normalize_box(box, width, height))
                    ner_tags.append(tag_id)
        
        all_words.append(words)
        all_boxes.append(boxes)
        all_labels.append(ner_tags)
        
    return {"words": all_words, "bboxes": all_boxes, "ner_tags": all_labels}

# --- 3. EXECUTION ---
print("1. Loading Dataset...")
dataset = load_dataset("naver-clova-ix/cord-v2")

print("2. Cleaning Data (This will remove the bad image)...")
# Applying the filter to remove the file causing OSError
dataset = dataset.filter(filter_corrupt_images)
print(f"   Cleaned Train Size: {len(dataset['train'])}")

print("3. Parsing OCR Data...")
processed_dataset = dataset.map(
    parse_cord_json,
    batched=True,
    batch_size=10, 
    remove_columns=["ground_truth"]
)

print("4. Preprocessing for LayoutLMv3...")
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

def prepare_batch(examples):
    return processor(
        examples['image'],
        examples['words'],
        boxes=examples['bboxes'],
        word_labels=examples['ner_tags'],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

encoded_dataset = processed_dataset.map(
    prepare_batch,
    batched=True,
    remove_columns=processed_dataset["train"].column_names
)
encoded_dataset.set_format(type="torch")

# --- 4. TRAINING SETUP ---
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(LABELS),
    label2id=label2id,
    id2label=id2label
)

metric = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    # Handle zero division warnings gracefully
    f1 = results.get("overall_f1", 0.0)
    acc = results.get("overall_accuracy", 0.0)
    return {"f1": f1, "accuracy": acc}

# Arguments (CPU SAFE)
args = TrainingArguments(
    output_dir="./cord_v2_model_final",
    max_steps=200,
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    logging_steps=10,
    load_best_model_at_end=True,
    fp16=False, 
    remove_unused_columns=False, 
    use_cpu=True # Fix 3: Force CPU usage
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
)

print("5. Starting Training...")
trainer.train()
trainer.save_model("./my_final_model")
processor.save_pretrained("./my_final_model")
print("Done! Model saved to ./my_final_model")