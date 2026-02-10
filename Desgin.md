1. Assumptions & Design Decisions
Input Assumptions
OCR Pre-processing: The prompt specifies "Input is OCR outputs." I assumed the service sits downstream from an OCR engine (like Tesseract or Google Vision). Therefore, the inference pipeline is designed to accept an image plus a list of words and bounding boxes, rather than raw pixel-only input.

Data Consistency: I assumed the CORD-v2 dataset schema (nested JSON with valid_line and quad coordinates) is the standard input format for this service.

Model Architecture: LayoutLMv3
I selected LayoutLMv3 (Microsoft) over standard text-only models (like BERT) or pure vision models (like ResNet).

Reasoning: Receipts are spatially dense. The meaning of a number (e.g., "$10.00") is defined by its position relative to other text (e.g., "Total" vs. "Tax"). LayoutLMv3 is multimodal: it simultaneously embeds Text, 2D Layout (Bounding Boxes), and Image Visuals. This makes it significantly more robust for document understanding than text-only approaches.

Data Processing Strategy
BIO Tagging: I utilized BIO (Begin-Inside-Outside) tagging for entity extraction. This is crucial for multi-token entities (e.g., "BURGER KING" becomes B-menu.nm, I-menu.nm) to ensure the downstream JSON reconstruction can correctly group words.

Coordinate Normalization: The model requires bounding boxes in a [0, 1000] scale. I implemented a robust normalize_box() function with clamping logic (max(0, min(1000, x))) to handle OCR noise where coordinates might slightly exceed image dimensions.

2. Evaluation Strategy
Primary Metric: F1-Score (SeqEval)
I chose F1-Score over Accuracy as the primary success metric.

Why: Document token classification is a highly imbalanced problem. The vast majority of tokens on a receipt are background text ("O" tag). A model could achieve 90% accuracy by predicting "O" for everything, but it would be useless. F1-Score (the harmonic mean of Precision and Recall) strictly penalizes missed entities, providing a true measure of extraction performance.

Robustness Checks
Data Integrity: During the data loading phase, I implemented a filter_corrupt_images() function. This scans the dataset for broken PNG headers or unreadable files (which were present in the source) and removes them before training, ensuring the pipeline is resilient to bad inputs.

3. Other Important Considerations
Hardware Optimization (Apple Silicon / CPU Fallback)
Training a multimodal Transformer is memory-intensive. My hardware (Apple M2 with Unified Memory), standard training loops caused Out of Memory crashes during the backward pass.

Solution: I optimized the training configuration by:

  Reducing Batch Size: Lowered per_device_train_batch_size to 2.

  Gradient Accumulation: Used gradient_accumulation_steps to simulate larger batch sizes without the memory overhead.

  CPU Offloading: Forced the trainer to use the CPU (use_cpu=True) to leverage system RAM (virtual memory) instead of crashing on the limited GPU memory.