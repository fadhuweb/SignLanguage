#!/usr/bin/env python
"""
Test script for the Sign Language Translation model.
This script verifies that the model can be loaded properly before starting the Flask application.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import cv2
from model import SignLanguageTranslator

def create_dummy_image():
    """Create a simple dummy image for testing model prediction."""
    # Create a black image with a white rectangle (simulating a hand gesture)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (200, 150), (400, 350), (255, 255, 255), -1)
    return img

def test_model(model_path):
    """
    Test loading and initialization of the sign language translation model.
    
    Args:
        model_path: Path to the model file
    
    Returns:
        True if model loads successfully, False otherwise
    """
    print(f"Testing model loading from: {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return False
    
    try:
        # Record start time for performance measurement
        start_time = time.time()
        
        # Detect device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Initialize the model
        translator = SignLanguageTranslator(model_path, device=device)
        
        # Calculate loading time
        loading_time = time.time() - start_time
        print(f"Model loaded successfully in {loading_time:.2f} seconds")
        
        # Print model information
        model_info = translator.get_model_info()
        print("\nModel Information:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
        
        # Try a simple prediction with a dummy image
        print("\nPerforming test prediction with dummy image...")
        dummy_img = create_dummy_image()
        
        # Time the prediction
        pred_start_time = time.time()
        translation, confidence = translator.translate_image(dummy_img)
        pred_time = time.time() - pred_start_time
        
        print(f"Test prediction completed in {pred_time:.2f} seconds")
        print(f"Sample translation: '{translation}'")
        print(f"Confidence: {confidence:.4f}")
        
        # Test the real-time processing functionality
        print("\nTesting real-time processing initialization...")
        translator.start_real_time_processing()
        time.sleep(1)  # Give it a moment to start
        translator.stop_real_time_processing()
        print("Real-time processing test completed")
        
        print("\nAll tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\nError during model testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to parse arguments and run the test."""
    parser = argparse.ArgumentParser(description="Test Sign Language Translation Model")
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/Downloads/best_slt_model.pt"),
                      help="Path to the model file")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Sign Language Translation Model Test")
    print("=" * 50)
    
    success = test_model(args.model)
    
    if success:
        print("\nModel verification completed successfully.")
        print("You can now start the Flask application.")
        sys.exit(0)
    else:
        print("\nModel verification failed.")
        print("Please check the error messages above and ensure the model file is correct.")
        sys.exit(1)

if __name__ == "__main__":
    main()

