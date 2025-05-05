# OCR Methods Comparison

## Tesseract

### Languages it supports
- Afrikaans, Amharic, Arabic, Assamese, Azerbaijani, Belarusian, Bengali, Bosnian, Bulgarian, Burmese, Catalan, Cebuano, 
  Central Khmer, Cherokee, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, 
  Esperanto, Estonian, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hebrew, Hindi, Hungarian, Icelandic, 
  Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Korean, Kurdish, Lao, Latin, Latvian, Lithuanian, 
  Macedonian, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Pashto, Persian, Polish, Portuguese, 
  Punjabi, Romanian, Russian, Serbian, Sinhala, Slovak, Slovenian, Spanish, Swahili, Swedish, Syriac, Tagalog, Tamil, 
  Telugu, Thai, Tibetan, Tigrinya, Turkish, Ukrainian, Urdu, Uzbek, Vietnamese, Welsh, Yiddish.

### Does it use GPU or only CPU?
- **CPU only**: Tesseract does not support GPU acceleration natively (no CUDA, no OpenCL support).

### License
- **Apache License 2.0**: Permissive open-source license, allowing commercial use, modification, distribution, and patent use.

### Processing speed
- **Standard Models**: On a modern CPU, it can process around 1–3 seconds per page.
- **Fast mode**: Use `--oem 0` engine mode for legacy OCR, which is faster but less accurate.

### Other important information
- **Page Segmentation Modes (PSM)**: 14 modes for layout analysis (e.g., single line, block, etc.)
- **Output formats**: Plain text, PDF, searchable PDF, hOCR, ALTO XML
- **Image input**: Works best with high-contrast, high-resolution (300 DPI+) images
- **Platform support**: Cross-platform (Linux, Windows, macOS)

---

## EasyOCR

### Languages it supports
- English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese (Simplified and Traditional), Japanese, Korean,
  Thai, Vietnamese, Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Marathi, Punjabi, Arabic, Hebrew,
  Persian, Urdu, Russian, Ukrainian, Bulgarian, Serbian, Greek, Georgian, Armenian.

### Does it use GPU or only CPU?
- **GPU Acceleration**: EasyOCR leverages PyTorch and can utilize CUDA-enabled GPUs for faster processing. To enable GPU
  usage, ensure that CUDA is properly installed and set `gpu=True` when initializing the Reader object.
- **CPU Usage**: If a compatible GPU is not available or `gpu=False` is set, EasyOCR will default to CPU processing.

### License
- **Apache License 2.0**: Permissive open-source license that allows for commercial use, modification, distribution, and patent use.

### Processing speed
- **GPU Performance**: When utilizing a GPU, EasyOCR can process images significantly faster.
- **CPU Performance**: Processing speed on CPU is slower and depends on factors such as image complexity and resolution. Users
  have reported that EasyOCR is notably slower on CPU, especially for high-resolution images.

### Other important information
- **Implementation**: EasyOCR is implemented in Python using the PyTorch deep learning framework.
- **Input Formats**: Supports various input formats, including image file paths, OpenCV images (NumPy arrays), image bytes,
  and image URLs.
- **Output Formats**: Returns results as a list of tuples containing the bounding box coordinates, detected text, and
  confidence score.
- **Customization**: Users can fine-tune EasyOCR models for specific use cases or to improve performance on custom datasets.
- **Integration**: EasyOCR can be integrated into various Python applications and frameworks, making it versatile for
  different OCR tasks.

---

## PaddleOCR

### Languages it supports
- English, French, Spanish, German, Italian, Portuguese, Dutch, Chinese (Simplified and Traditional), Japanese, Korean,
  Thai, Vietnamese, Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Marathi, Punjabi, Arabic, Hebrew,
  Persian, Urdu, Russian, Ukrainian, Bulgarian, Serbian, Greek, Georgian, Armenian.

### Does it use GPU or only CPU?
- **GPU Acceleration**: PaddleOCR leverages PaddlePaddle and can utilize CUDA-enabled GPUs for faster processing. To enable
  GPU usage, ensure that CUDA is properly installed and set `use_gpu=True` when initializing the PaddleOCR object.
- **CPU Usage**: If a compatible GPU is not available or `use_gpu=False` is set, PaddleOCR will default to CPU processing. Note
  that OCR operations will be slower on CPU compared to GPU.

### License
- **Apache License 2.0**: Permissive open-source license that allows for commercial use, modification, distribution, and patent use.

### Processing speed
- **GPU Performance**: When utilizing a GPU, PaddleOCR can process images significantly faster.
- **CPU Performance**: Processing speed on CPU is slower and depends on factors such as image complexity and resolution. Users
  have reported that PaddleOCR is notably slower on CPU, especially for high-resolution images.

### Other important information
- **Implementation**: PaddleOCR is implemented in Python using the PaddlePaddle deep learning framework.
- **Input Formats**: Supports various input formats, including image file paths, OpenCV images (NumPy arrays), image bytes,
  and image URLs.
- **Output Formats**: Returns results as a list of tuples containing the bounding box coordinates, detected text, and
  confidence score.
- **Customization**: Users can fine-tune PaddleOCR models for specific use cases or to improve performance on custom datasets.
- **Integration**: PaddleOCR can be integrated into various Python applications and frameworks, making it versatile for
  different OCR tasks.

---

## Google Cloud Vision OCR

### Languages it supports
- English, Spanish, French, German, Italian, Chinese (Simplified and Traditional), Japanese, Korean, Thai, Vietnamese,
  Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Russian, Ukrainian, Serbian, Arabic, Hebrew, Urdu, many
  others.

### Does it use GPU or only CPU?
- **Cloud-based**: Runs on Google’s infrastructure; you don’t manage CPU/GPU directly. Ideal for applications where local
  hardware is a limitation or scalability is important.

### License
- **Commercial service**: No open-source license applies. Usage is governed by Google Cloud's Terms of Service.

### Processing speed
- Extremely fast due to cloud infrastructure. Can handle batch processing and large documents efficiently. Depends on your
  internet connection and API usage limits.

### Other important information
- **Highly accurate**, especially for printed documents. Supports document layout detection, such as columns and tables.
- Can extract **handwritten text** with reasonable accuracy.
- Integrates with other **Google Cloud services** (e.g., AutoML, Document AI).
- Requires an **API key** and is not free beyond the free-tier quota.
