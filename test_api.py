#!/usr/bin/env python3
"""
Test script for the /api/translate-sequence endpoint of the Sign Language Translation API.

This script generates a simple test image, encodes it to base64, and sends it to the API
as part of a frames array to test the translate-sequence functionality.
"""

import base64
import io
import json
import requests
from PIL import Image, ImageDraw
import time
import sys

# API endpoint URL
API_URL = "http://localhost:5000/api/translate-sequence"

def create_test_image(width=224, height=224, color=(255, 0, 0), text="Test"):
    """
    Create a simple test image with solid color background and optional text.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        color: RGB color tuple for background
        text: Text to display on the image
        
    Returns:
        PIL Image object
    """
    image = Image.new('RGB', (width, height), color=color)
    
    # Add text if provided
    if text:
        draw = ImageDraw.Draw(image)
        # Use a simple font and size
        draw.text((width//4, height//2), text, fill=(255, 255, 255))
    
    return image

def encode_image_to_base64(image):
    """
    Convert a PIL Image to base64 encoded string.
    
    Args:
        image: PIL Image object
        
    Returns:
        base64 encoded string
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def test_api_with_single_frame():
    """
    Test the API with a single frame image.
    """
    print("Testing /api/translate-sequence with a single frame...")
    
    try:
        # Create and encode a test image
        image = create_test_image()
        encoded_image = encode_image_to_base64(image)
        
        # Prepare the payload with the encoded image in the frames array
        payload = {
            "frames": [encoded_image]
        }
        
        # Make the API request
        print(f"Sending request to {API_URL}...")
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        end_time = time.time()
        
        # Print response details
        print(f"Request completed in {end_time - start_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("\nResponse:")
                print(json.dumps(result, indent=4))
                print(f"\nTranslation: {result.get('translation', 'No translation')}")
                print(f"Confidence: {result.get('confidence', 0)}")
            except json.JSONDecodeError:
                print("Error: Unable to parse JSON response")
                print(f"Raw response: {response.text}")
        else:
            print(f"Error: {response.status_code} - {response.reason}")
            print(f"Response body: {response.text}")
            
    except requests.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def test_api_with_multiple_frames(num_frames=5):
    """
    Test the API with multiple frames.
    
    Args:
        num_frames: Number of frames to generate
    """
    print(f"\nTesting /api/translate-sequence with {num_frames} frames...")
    
    try:
        # Create and encode multiple test images with different colors
        encoded_images = []
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        
        for i in range(num_frames):
            color_idx = i % len(colors)
            image = create_test_image(color=colors[color_idx], text=f"Frame {i+1}")
            encoded_image = encode_image_to_base64(image)
            encoded_images.append(encoded_image)
        
        # Prepare the payload with the encoded images in the frames array
        payload = {
            "frames": encoded_images
        }
        
        # Make the API request
        print(f"Sending request to {API_URL}...")
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        end_time = time.time()
        
        # Print response details
        print(f"Request completed in {end_time - start_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("\nResponse:")
                print(json.dumps(result, indent=4))
                print(f"\nTranslation: {result.get('translation', 'No translation')}")
                print(f"Confidence: {result.get('confidence', 0)}")
            except json.JSONDecodeError:
                print("Error: Unable to parse JSON response")
                print(f"Raw response: {response.text}")
        else:
            print(f"Error: {response.status_code} - {response.reason}")
            print(f"Response body: {response.text}")
            
    except requests.RequestException as e:
        print(f"Request error: {e}")
    except Exception as e:
        print(f"Error: {e}")

def check_api_health():
    """
    Check if the API is running by calling the health endpoint.
    """
    print("Checking API health...")
    try:
        response = requests.get("http://localhost:5000/api/health")
        if response.status_code == 200:
            result = response.json()
            print(f"API is healthy: {result}")
            print(f"Model loaded: {result.get('model_loaded', False)}")
            return True
        else:
            print(f"API health check failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"Error connecting to API: {e}")
        return False

if __name__ == "__main__":
    # First check if the API is running
    if check_api_health():
        # Test with a single frame
        test_api_with_single_frame()
        
        # Test with multiple frames
        test_api_with_multiple_frames(5)
    else:
        print("API health check failed. Make sure the API server is running.")
        sys.exit(1)

#!/usr/bin/env python
"""
Test script for the Sign Language Translation API.
This script verifies that the Flask API can start up and respond to requests.
"""

import os
import sys
import time
import argparse
import requests
import subprocess
import threading
import socket
import json
from urllib.parse import urljoin

class APITester:
    """Class to test the Flask API for the Sign Language Translation app."""
    
    def __init__(self, host='127.0.0.1', port=5000, timeout=30):
        """
        Initialize the API tester.
        
        Args:
            host: Host where the API is running
            port: Port where the API is running
            timeout: Timeout in seconds for API to become available
        """
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.process = None
    
    def is_port_in_use(self):
        """Check if the port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((self.host, self.port)) == 0
    
    def start_api(self):
        """Start the Flask API in a subprocess if it's not already running."""
        if self.is_port_in_use():
            print(f"Port {self.port} is already in use. Assuming API is running.")
            return True
        
        try:
            # Start the Flask app in a subprocess
            print(f"Starting Flask API on {self.base_url}...")
            self.process = subprocess.Popen(
                [sys.executable, "app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for the API to start
            start_time = time.time()
            while not self.is_port_in_use():
                if time.time() - start_time > self.timeout:
                    print(f"Timeout waiting for API to start after {self.timeout} seconds")
                    self.stop_api()
                    return False
                
                # Check if process is still running
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    print(f"API process terminated with exit code {self.process.returncode}")
                    print(f"STDOUT: {stdout}")
                    print(f"STDERR: {stderr}")
                    return False
                
                time.sleep(0.5)
                
            print(f"Flask API started on {self.base_url}")
            return True
            
        except Exception as e:
            print(f"Error starting API: {str(e)}")
            if self.process:
                self.stop_api()
            return False
    
    def stop_api(self):
        """Stop the API if it was started by this script."""
        if self.process:
            print("Stopping Flask API...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
    
    def test_health_endpoint(self):
        """Test the health check endpoint of the API."""
        endpoint = "/api/health"
        url = urljoin(self.base_url, endpoint)
        
        try:
            print(f"Testing health endpoint: {url}")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Health check successful: {data}")
                
                # Check if model is loaded
                if data.get('model_loaded', False):
                    print("Model is loaded correctly")
                else:
                    print("WARNING: Model is not loaded")
                    
                return True
            else:
                print(f"Health check failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.RequestException as e:
            print(f"Error connecting to API: {str(e)}")
            return False
    
    def test_model_info_endpoint(self):
        """Test the model info endpoint of the API."""
        endpoint = "/api/model-info"
        url = urljoin(self.base_url, endpoint)
        
        try:
            print(f"Testing model info endpoint: {url}")
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print("Model information:")
                for key, value in data.items():
                    print(f"  {key}: {value}")
                return True
            else:
                print(f"Model info request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.RequestException as e:
            print(f"Error connecting to API: {str(e)}")
            return False
    
    def test_sample_translation(self):
        """Test the translation endpoint with a sample image."""
        import cv2
        import base64
        
        # Create a dummy image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(img, (200, 150), (400, 350), (255, 255, 255), -1)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare the request
        endpoint = "/api/translate"
        url = urljoin(self.base_url, endpoint)
        payload = {"image": img_base64}
        
        try:
            print(f"Testing translation endpoint with sample image: {url}")
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"Translation result: {data.get('translation')}")
                print(f"Confidence: {data.get('confidence')}")
                return True
            else:
                print(f"Translation request failed with status code: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except requests.RequestException as e:
            print(f"Error connecting to API: {str(e)}")
            return False

def main():
    """Main function to parse arguments and run the API tests."""
    parser = argparse.ArgumentParser(description="Test Sign Language Translation API")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                      help="Host where the API is running")
    parser.add_argument("--port", type=int, default=5000,
                      help="Port where the API is running")
    parser.add_argument("--no-start", action="store_true",
                      help="Don't try to start the API, just test it")
    parser.add_argument("--sample-translation", action="store_true",
                      help="Test the translation endpoint with a sample image")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Sign Language Translation API Test")
    print("=" * 50)
    
    tester = APITester(host=args.host, port=args.port)
    
    try:
        # Start the API if requested
        if not args.no_start:
            if not tester.start_api():
                print("Failed to start API. Exiting tests.")
                sys.exit(1)
        
        # Wait a moment for the API to fully initialize
        time.sleep(2)
        
        # Test the health endpoint
        health_ok = tester.test_health_endpoint()
        
        # Test the model info endpoint
        info_ok = tester.test_model_info_endpoint()
        
        # Optionally test the translation endpoint
        translation_ok = True
        if args.sample_translation:
            import numpy as np
            import cv2
            translation_ok = tester.test_sample_translation()
        
        # Summarize results
        print("\nTest Results:")
        print(f"Health Endpoint: {'PASS' if health_ok else 'FAIL'}")
        print(f"Model Info Endpoint: {'PASS' if info_ok else 'FAIL'}")
        if args.sample_translation:
            print(f"Sample Translation: {'PASS' if translation_ok else 'FAIL'}")
        
        all_passed = health_ok and info_ok and translation_ok
        
        if all_passed:
            print("\nAll API tests passed successfully!")
            print("The Flask API is running correctly and can be used for frontend development.")
            sys.exit(0)
        else:
            print("\nSome API tests failed.")
            print("Please check the error messages above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTests interrupted by user.")
    finally:
        # Stop the API if we started it
        if not args.no_start:
            tester.stop_api()

if __name__ == "__main__":
    main()

