<<<<<<< HEAD
# Sign Language Translation Application

## Introduction

This application provides real-time translation of sign language from images and videos into text. It uses a deep learning model that combines DINOv2 for visual feature extraction, LSTM for temporal processing, and BART for text generation to accurately translate sign language gestures.

Key features:
- Upload and translate images containing sign language gestures
- Upload and translate videos of sign language sequences
- Simple web interface for easy interaction
- RESTful API for programmatic integration
- Support for multiple file formats

The application is built with Flask (Python) for the backend and uses a simple HTML/JavaScript frontend for user interaction.

## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.8 or higher
- OpenCV
- Flask
- A pre-trained sign language translation model

### Installation Steps

1. Clone the repository or download the application files:

```
git clone https://github.com/yourusername/SignLanguageApp.git
cd SignLanguageApp
```

2. Create and activate a virtual environment:

```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install torch torchvision transformers opencv-python flask flask-cors pillow nltk
```

4. Download the pre-trained model and place it in the appropriate location:
   - The default location is `~/Downloads/best_slt_model.pt`
   - You can change this path in `app.py` by modifying the `MODEL_PATH` variable

## Starting the Flask Server

1. Navigate to the backend directory:

```bash
cd backend
```

2. Start the Flask application:

```bash
# On Windows
python app.py

# On macOS/Linux
python3 app.py
```

3. The server will start and display a message like:
```
Model loaded successfully on cpu
 * Running on http://0.0.0.0:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 123-456-789
```

4. The application is now running and ready to accept requests.

## Accessing and Using the Web Interface

1. Open a web browser and navigate to:
```
http://localhost:5000/
```

2. You should see the Sign Language Translator interface with tabs for Image Upload, Video Upload, and API Info.

### Uploading and Translating Images

1. Click on the "Image Upload" tab (selected by default)
2. Click the "Choose File" button and select an image file containing sign language gestures
   - Supported formats: JPG, JPEG, PNG, GIF, BMP
   - Maximum file size: 16 MB
3. Once an image is selected, a preview will appear
4. Click the "Translate Image" button
5. The translation will appear in the "Translation Results" section on the right

### Uploading and Translating Videos

1. Click on the "Video Upload" tab
2. Click the "Choose File" button and select a video file containing sign language gestures
   - Supported formats: MP4, AVI, MOV, WMV, WebM
   - Maximum file size: 16 MB
3. Adjust the "Maximum Frames to Process" value if needed
   - Higher values may improve accuracy but take longer to process
   - Default: 30 frames
4. Once a video is selected, a preview will appear
5. Click the "Translate Video" button
6. The translation will appear in the "Translation Results" section on the right

### Viewing API Information

1. Click on the "API Info" tab
2. This tab displays information about:
   - Allowed file types for upload
   - Maximum file size
   - Model information and configuration



## Troubleshooting

### Common Issues and Solutions

1. **Model fails to load**
   - Ensure the model file exists at the specified path
   - Check if you have enough memory (RAM) to load the model
   - Verify that you have the correct PyTorch version installed

2. **"No module found" errors**
   - Make sure you've installed all required dependencies
   - Check that your virtual environment is activated

3. **File upload issues**
   - Ensure your file is one of the supported formats
   - Check that the file size is under 16 MB
   - Try a different file to see if the issue persists

4. **Cross-Origin (CORS) errors in browser console**
   - This can happen if you're accessing the API from a different domain
   - The API has CORS enabled, but you might need to adjust settings if accessing from a custom domain

5. **Slow processing for videos**
   - Try reducing the "Maximum Frames to Process" value
   - Consider using a smaller video file
   - If available, use a system with GPU support for faster processing






=======
# SignLanguage
>>>>>>> e69c0eed0f95b2ca7bcf2ab4e02ed5bd82c329ae
