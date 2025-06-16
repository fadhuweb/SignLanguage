import os
import cv2
import numpy as np
import torch
import transformers
import nltk
import nltk.translate.bleu_score
from typing import List, Tuple, Dict, Any, Optional
from torchvision import transforms
from PIL import Image
import queue
import threading
import time

class SignLanguageTranslationModel(torch.nn.Module):
    def __init__(
            self,
            dino_model_name     = 'facebook/dinov2-base',
            bart_model_name     = 'facebook/bart-base',
            hidden_dim          = 768,
            lstm_hidden_dim     = 512,
            num_lstm_layers     = 1,
            dropout             = 0.1,
            freeze_dino         = True,
            max_length          = 50
    ):
        super().__init__()

        # DINOv2 for visual feature extraction
        self.dino = transformers.Dinov2Model.from_pretrained(dino_model_name)
        if freeze_dino:
            for param in self.dino.parameters():
                param.requires_grad = False

        # Linear projection
        self.projection = torch.nn.Linear(self.dino.config.hidden_size, hidden_dim)
        self.layer_norm = torch.nn.LayerNorm(hidden_dim)

        # LSTM for temporal aggregation
        self.lstm = torch.nn.LSTM(
            input_size          = hidden_dim,
            hidden_size         = lstm_hidden_dim,
            num_layers          = num_lstm_layers,
            batch_first         = True,
            bidirectional       = True,
            dropout             = dropout if num_lstm_layers > 1 else 0
        )

        # BART for text generation
        self.bart_config = transformers.BartConfig.from_pretrained(bart_model_name)
        self.bart = transformers.BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.bart_tokenizer = transformers.BartTokenizer.from_pretrained(bart_model_name)

        # Linear layer connecting LSTM and BART
        self.lstm_to_bart = torch.nn.Linear(lstm_hidden_dim * 2, self.bart_config.d_model)

        self.max_length = max_length

    def temporal_downsampling(self, x, factor=3, target_length=50):
        _, seq_len, _ = x.size()
        x = x.transpose(1, 2)
        x = torch.nn.functional.adaptive_avg_pool1d(x, seq_len // factor)
        x = x.transpose(1, 2)

        # Pad or truncate to target_length
        current_length = x.size(1)
        if current_length < target_length:
            padding = torch.zeros(x.size(0), target_length - current_length, x.size(2), device=x.device)
            x = torch.cat([x, padding], dim=1)
        elif current_length > target_length:
            x = x[:, :target_length, :]
        
        return x

    def encode(self, frames, attention_mask=None):
        batch_size, num_frames, channels, height, width = frames.size()

        # reshape for DINOv2 processing
        frames = frames.view(batch_size * num_frames, channels, height, width)

        # extract features using DINOv2
        with torch.no_grad() if self.dino.training == False else torch.enable_grad():
            dino_output = self.dino(frames)
            visual_features = dino_output.last_hidden_state[:, 0, :]  # use [CLS]

        # reshape back: [batch_size, num_features, features]
        visual_features = visual_features.view(batch_size, num_frames, -1)

        # apply projection and normalization
        projected_features = self.projection(visual_features)
        normalized_features = self.layer_norm(projected_features)

        # do downsampling
        downsampled_features = self.temporal_downsampling(normalized_features, target_length=self.max_length)

        # apply attention mask if possible
        if attention_mask is not None:
            downsampled_mask = attention_mask[:, ::3]
            
            target_length = self.max_length
            current_length = downsampled_mask.size(1)
            if current_length < target_length:
                padding = torch.zeros(downsampled_mask.size(0), target_length - current_length, device=downsampled_mask.device)
                downsampled_mask = torch.cat([downsampled_mask, padding], dim=1)
            elif current_length > target_length:
                downsampled_mask = downsampled_mask[:, :target_length]
            
            downsampled_features = downsampled_features * downsampled_mask.unsqueeze(-1)

        # run LSTM
        lstm_output, _ = self.lstm(downsampled_features)

        # output for BART's expected input
        bart_features = self.lstm_to_bart(lstm_output)

        return bart_features
    
    def decode(self, encoded_features, target_ids=None):
        
        if target_ids is not None:
            # training mode
            # Teacher forcing approach
            decoder_input_ids = target_ids.clone()
            decoder_input_ids[:, :-1] = target_ids[:, 1:]
            decoder_input_ids[:, -1] = self.bart_tokenizer.pad_token_id
            
            decoder_output = self.bart(
                encoder_outputs = [encoded_features],
                # decoder_input_ids = decoder_input_ids,
                labels = target_ids,
                return_dict = True
            )
            return {
                'loss': decoder_output.loss,                # cross-entropy loss
                'logits': decoder_output.logits
            }

        # inference mode
        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=encoded_features)
        generated_ids = self.bart.generate(
            encoder_outputs = encoder_outputs,
            max_length = self.max_length,
            num_beams = 4,
            early_stopping = True,
            repetition_penalty = 1.2,
            no_repeat_ngram_size = 3
        )
        return {
            'generated_ids': generated_ids
        }
    
    def forward(self, frames, attention_mask=None, target_ids=None):
        encoded_features = self.encode(frames, attention_mask)
        output = self.decode(encoded_features, target_ids)
        return output


import torch
import os
import cv2
import queue
import nltk
import transformers
import numpy as np
from typing import List, Optional
from PIL import Image
from torchvision import transforms
 # Ensure your model is correctly imported


class SignLanguageTranslator:
    """
    A class for loading and using a PyTorch sign language translation model.
    This class provides methods for translating sign language from images or video frames
    and supports real-time two-way communication between deaf and hearing people.
    """

    def __init__(self, model_path: str, device: Optional[torch.device] = None,
                 frame_buffer_size: int = 30, translation_threshold: float = 0.6):
        """
        Initialize the sign language translator with a pre-trained model.

        Args:
            model_path: Path to the saved PyTorch model (.pt file)
            device: The device to run the model on (CPU or CUDA). If None, will use CUDA if available.
            frame_buffer_size: Number of frames to buffer for continuous sign language detection
            translation_threshold: Confidence threshold for considering a translation valid
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        transformers.logging.set_verbosity_error()

        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        print(f"Loading model from {model_path}...")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = self.model.bart_tokenizer

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.frame_buffer_size = frame_buffer_size
        self.translation_threshold = translation_threshold
        self.frame_buffer = []
        self.last_translation = ""
        self.last_translation_time = 0
        self.translation_cooldown = 2.0

        self.processing_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()
        self.is_running = False
        self.processing_thread = None

        self._model_info = {
            "name": "Sign Language Translation Model",
            "type": "PyTorch",
            "architecture": "DINOv2-LSTM-BART",
            "input_shape": [3, 224, 224],
            "output_type": "text",
            "model_path": model_path,
            "device": str(self.device),
            "frame_buffer_size": frame_buffer_size,
            "translation_threshold": translation_threshold
        }

    def translate_sequence(self, frames: List[np.ndarray]) -> str:
        """
        Translate a sequence of video frames into a natural language sentence.

        Args:
            frames: List of RGB images (np.ndarray) representing a video clip.

        Returns:
            A string containing the predicted sentence.
        """
        processed_frames = []
        for frame in frames:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = self.transform(img).unsqueeze(0)
            processed_frames.append(img)

        input_tensor = torch.cat(processed_frames, dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(input_tensor)
            translation = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return translation

    def _load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load the PyTorch model from the specified path.

        Args:
            model_path: Path to the saved PyTorch model (.pt file)

        Returns:
            The loaded PyTorch model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            model_data = torch.load(model_path, map_location=self.device, weights_only=False)

            if isinstance(model_data, dict):
                if "model_state_dict" in model_data:
                    model = SignLanguageTranslationModel(
                        dino_model_name=model_data.get("dino_model_name", "facebook/dinov2-base"),
                        bart_model_name=model_data.get("bart_model_name", "facebook/bart-base"),
                        hidden_dim=model_data.get("hidden_dim", 768),
                        lstm_hidden_dim=model_data.get("lstm_hidden_dim", 512),
                        num_lstm_layers=model_data.get("num_lstm_layers", 1),
                        dropout=model_data.get("dropout", 0.1),
                        freeze_dino=True,
                        max_length=model_data.get("max_length", 50)
                    )
                    model.load_state_dict(model_data["model_state_dict"])
                    return model
                else:
                    try:
                        model = SignLanguageTranslationModel(
                            dino_model_name="facebook/dinov2-base",
                            bart_model_name="facebook/bart-base",
                            hidden_dim=768,
                            lstm_hidden_dim=512,
                            num_lstm_layers=1,
                            dropout=0.1,
                            freeze_dino=True,
                            max_length=50
                        )
                        model.load_state_dict(model_data)
                        print("Successfully loaded state dictionary into a new model instance")
                        return model
                    except Exception as state_dict_error:
                        print(f"Failed to load as state dict: {str(state_dict_error)}")
                        if hasattr(model_data, 'to') and callable(getattr(model_data, 'to')):
                            return model_data
                        else:
                            raise RuntimeError("Loaded object is not a valid PyTorch model.")
            else:
                return model_data

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single image for the model.
        
        Args:
            image: OpenCV image (numpy array, BGR format)
            
        Returns:
            Preprocessed image tensor ready for the model
        """
        # Convert BGR to RGB (OpenCV uses BGR by default)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transformations
        image_tensor = self.transform(pil_image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess a sequence of frames for the model.
        
        Args:
            frames: List of OpenCV images (numpy arrays, BGR format)
            
        Returns:
            Preprocessed tensor with shape [1, num_frames, 3, height, width]
        """
        if not frames:
            return None
            
        processed_frames = []
        
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_frame = Image.fromarray(frame_rgb)
            
            # Apply transformations
            frame_tensor = self.transform(pil_frame)
            
            processed_frames.append(frame_tensor)
        
        # Stack all frames
        frames_tensor = torch.stack(processed_frames, dim=0)
        
        # Add batch dimension: [1, num_frames, 3, height, width]
        frames_tensor = frames_tensor.unsqueeze(0)
        
        return frames_tensor.to(self.device)
    
    def translate_image(self, image: np.ndarray) -> str:
        """
        Translate sign language from a single image.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            The translated text
        """
        # For a single image, we'll treat it as a sequence of 1 frame
        image_tensor = self.preprocess_image(image)
        
        # Reshape to match the expected input: [batch_size, num_frames, channels, height, width]
        image_tensor = image_tensor.view(1, 1, 3, 224, 224)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
        
        # Process the outputs to get the translated text
        translation = self._process_model_output(outputs)
        
        return translation
    
    def translate_sequence(self, frames: List[np.ndarray]) -> str:
        """
        Translate sign language from a sequence of frames.
        
        Args:
            frames: List of OpenCV images (numpy arrays)
            
        Returns:
            The complete translated text
        """
        if not frames:
            return ""
        
        # Preprocess frames
        frames_tensor = self.preprocess_frames(frames)
        
        if frames_tensor is None:
            return ""
            
        # Run inference
        with torch.no_grad():
            outputs = self.model(frames_tensor)
        
        # Process the outputs to get the translated text
        translation = self._process_model_output(outputs)
        
        return translation
    
    def _process_model_output(self, outputs: Dict[str, torch.Tensor]) -> str:
        """
        Process the raw model output to get translated text.
        
        Args:
            outputs: Raw output dictionary from the model
            
        Returns:
            The translated text
        """
        # Extract the generated IDs from the output
        if 'generated_ids' in outputs:
            generated_ids = outputs['generated_ids']
            
            # Decode the IDs to text
            translation = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            return translation        
        return "No translation available"
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            A dictionary containing model information
        """
        return self._model_info
        
    def add_frame_to_buffer(self, frame: np.ndarray) -> None:
        """
        Add a frame to the buffer for real-time processing.
        
        Args:
            frame: OpenCV image (numpy array, BGR format)
        """
        # Add frame to buffer
        self.frame_buffer.append(frame.copy())
        
        # Keep buffer size within limits
        if len(self.frame_buffer) > self.frame_buffer_size:
            self.frame_buffer.pop(0)
            
        # If we're running real-time processing, add to the queue
        if self.is_running and not self.processing_queue.full():
            self.processing_queue.put(frame.copy())
            
    def start_real_time_processing(self):
        """
        Start real-time processing in a separate thread.
        """
        if self.is_running:
            return
            
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("Real-time sign language processing started")
        
    def stop_real_time_processing(self):
        """
        Stop the real-time processing thread.
        """
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
        print("Real-time sign language processing stopped")
        
    def _processing_worker(self):
        """
        Worker function for real-time processing thread.
        """
        frame_sequence = []
        last_translation_time = 0
        
        while self.is_running:
            try:
                # Get a frame from the queue with timeout
                try:
                    frame = self.processing_queue.get(timeout=0.1)
                    frame_sequence.append(frame.copy())
                    
                    # Keep sequence within limits
                    if len(frame_sequence) > self.frame_buffer_size:
                        frame_sequence.pop(0)
                except queue.Empty:
                    # No new frame, continue processing current sequence
                    pass
                
                # Process sequence if we have enough frames and enough time has passed
                current_time = time.time()
                if (len(frame_sequence) >= 10 and 
                    current_time - last_translation_time > self.translation_cooldown):
                    
                    # Translate the sequence
                    translation = self.translate_sequence(frame_sequence)
                    
                    # Use a default confidence value
                    confidence = 0.8
                    
                    # If translation exists and isn't empty
                    if translation:
                        self.result_queue.put((translation, confidence, current_time))
                        last_translation_time = current_time
                        frame_sequence = []  # Reset sequence after translation
                        
            except Exception as e:
                print(f"Error in processing worker: {str(e)}")
                time.sleep(0.1)  # Prevent tight loop on error
                
    def get_latest_translation(self) -> Tuple[str, float, float]:
        """
        Get the latest translation from the result queue.
        
        Returns:
            A tuple containing:
                - The translated text
                - The confidence score (0-1)
                - The timestamp of the translation
                
            If no new translation is available, returns (None, 0.0, 0.0)
        """
        try:
            translation, confidence, timestamp = self.result_queue.get_nowait()
            return translation, confidence, timestamp
        except queue.Empty:
            return None, 0.0, 0.0

    def evaluate_model(self, test_loader, calc_bleu=True, num_beams=5, max_length=60):
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_loader: DataLoader containing test data
            calc_bleu: Whether to calculate BLEU score
            num_beams: Number of beams for beam search
            max_length: Maximum length of generated sequences
            
        Returns:
            Dictionary with evaluation results
        """
        self.model.eval()

        predictions = []
        references = []

        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for batch in test_loader:
                frames = batch['frames'].to(self.device)
                frames_mask = batch['frames_mask'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)

                # Compute loss (teacher forcing)
                outputs = self.model(frames, frames_mask, input_ids)
                loss = outputs['loss']
                total_loss += loss.item()
                batches += 1

                # Inference
                encoded = self.model.encode(frames, frames_mask)
                decode_outputs = self.model.decode(encoded)
                generated_ids = decode_outputs['generated_ids']

                preds = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                refs = batch['captions']  # raw strings

                predictions.extend(preds)
                references.extend(refs)

        avg_loss = total_loss / batches

        results = {
            'test_loss': avg_loss,
            'predictions': predictions,
            'references': references
        }

        if calc_bleu:
            tokenized_preds = [nltk.word_tokenize(p.lower().strip()) for p in predictions]
            tokenized_refs = [[nltk.word_tokenize(r.lower().strip())] for r in references]

            smoothing = nltk.translate.bleu_score.SmoothingFunction().method1
            score = nltk.translate.bleu_score.corpus_bleu(
                tokenized_refs, tokenized_preds, smoothing_function=smoothing
            )
            results['bleu_score'] = score
            print(f'=== BLEU Score: {score * 100:.2f} ===')

        return results

