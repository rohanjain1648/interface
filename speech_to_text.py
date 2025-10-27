import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
from groq import Groq
import pyaudio
import numpy as np
import threading
import time
from adhd_detection import ADHDDetector

from dotenv import load_dotenv  # Import load_dotenv

load_dotenv()  # Load environment variables from .env file
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Function to record audio from the microphone and save it as an MP3 file.

    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_lfimit (int): Maximum time for the phrase to be recorded (in seconds).
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            
            # Record the audio
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            # Convert the recorded audio to an MP3 file
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            
            logging.info(f"Audio saved to {file_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

# filepath = "test_speech_to_text.mp3"
# record_audio(filepath)



def transcribe_with_groq(audio_filepath):
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set.")

    client = Groq() # The Groq client automatically picks up the API key from the environment
    stt_model="whisper-large-v3"
    audio_file=open(audio_filepath, "rb")
    transcription=client.audio.transcriptions.create(
        model=stt_model,
        file=audio_file,
        language="en"
    )

    return transcription.text

class ContinuousSpeechRecognizer:
    def __init__(self, adhd_detector=None):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.adhd_detector = adhd_detector
        self.is_listening = False
        self.thread = None
        
        # Question monitoring state
        self.question_monitoring_active = False
        self.current_question_start_time = None
        self.current_question_completion_time = None
        self.interruption_callback = None
        self.response_callback = None
        
        # Interruption detection parameters
        self.interruption_detection_delay = 0.5  # Minimum delay to consider as interruption (seconds)
        self.response_timeout = 10  # Maximum time to wait for response after question completion

    def start_continuous_listening(self, callback):
        """Start continuous listening in a separate thread"""
        self.is_listening = True
        self.thread = threading.Thread(target=self._listen_loop, args=(callback,))
        self.thread.start()

    def stop_continuous_listening(self):
        """Stop continuous listening"""
        self.is_listening = False
        if self.thread:
            self.thread.join()

    def _listen_loop(self, callback):
        """Main listening loop with interruption detection"""
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Continuous listening started...")

            while self.is_listening:
                try:
                    logging.info("Listening for speech...")
                    
                    # Record the time when speech is detected
                    speech_detection_time = time.time()
                    
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    logging.info("Speech detected, processing...")

                    # Save audio to file for transcription
                    wav_data = audio.get_wav_data()
                    audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
                    temp_filepath = "temp_audio.mp3"
                    audio_segment.export(temp_filepath, format="mp3", bitrate="128k")

                    # Transcribe
                    transcription = transcribe_with_groq(temp_filepath)

                    # Check for interruption if question monitoring is active
                    if self.question_monitoring_active:
                        self._handle_speech_during_question_monitoring(transcription, speech_detection_time)
                    else:
                        # Call normal callback with transcription
                        callback(transcription)

                    # Clean up
                    os.remove(temp_filepath)

                except sr.WaitTimeoutError:
                    continue  # No speech detected, continue listening
                except Exception as e:
                    logging.error(f"Error in listening loop: {e}")
                    time.sleep(1)  # Brief pause before retrying

    def start_question_monitoring(self, question_start_time=None, interruption_callback=None, response_callback=None):
        """
        Start monitoring for interruptions during question presentation.
        
        Args:
            question_start_time: Time when question presentation started (defaults to current time)
            interruption_callback: Function to call when interruption is detected
            response_callback: Function to call for valid responses
        """
        self.question_monitoring_active = True
        self.current_question_start_time = question_start_time or time.time()
        self.current_question_completion_time = None
        self.interruption_callback = interruption_callback
        self.response_callback = response_callback
        
        logging.info(f"Question monitoring started at {self.current_question_start_time}")

    def mark_question_completion(self, completion_time=None):
        """
        Mark that the current question presentation has completed.
        
        Args:
            completion_time: Time when question presentation completed (defaults to current time)
        """
        if not self.question_monitoring_active:
            logging.warning("Attempted to mark question completion when monitoring not active")
            return
            
        self.current_question_completion_time = completion_time or time.time()
        logging.info(f"Question presentation completed at {self.current_question_completion_time}")

    def stop_question_monitoring(self):
        """Stop monitoring for interruptions and reset question state."""
        self.question_monitoring_active = False
        self.current_question_start_time = None
        self.current_question_completion_time = None
        self.interruption_callback = None
        self.response_callback = None
        
        logging.info("Question monitoring stopped")

    def _handle_speech_during_question_monitoring(self, transcription, speech_detection_time):
        """
        Handle speech detected during question monitoring to determine if it's an interruption or valid response.
        
        Args:
            transcription: The transcribed speech text
            speech_detection_time: Time when speech was first detected
        """
        if not self.question_monitoring_active:
            return

        # Determine if this is an interruption or valid response
        is_interruption = self._is_speech_interruption(speech_detection_time)
        
        if is_interruption:
            self._handle_interruption(transcription, speech_detection_time)
        else:
            self._handle_valid_response(transcription, speech_detection_time)

    def _is_speech_interruption(self, speech_detection_time):
        """
        Determine if speech detected at given time constitutes an interruption.
        
        Args:
            speech_detection_time: Time when speech was detected
            
        Returns:
            bool: True if this is considered an interruption, False otherwise
        """
        if not self.current_question_start_time:
            return False
            
        # If question presentation hasn't completed yet, any speech is an interruption
        if self.current_question_completion_time is None:
            # Check if enough time has passed since question start to avoid false positives
            time_since_start = speech_detection_time - self.current_question_start_time
            return time_since_start >= self.interruption_detection_delay
        
        # If question has completed, check if response came too quickly (might be interruption)
        # or if it's a valid response after completion
        time_since_completion = speech_detection_time - self.current_question_completion_time
        
        # If response comes very quickly after completion, it might have been an interruption
        # that was processed after the question finished
        return time_since_completion < self.interruption_detection_delay

    def _handle_interruption(self, transcription, interruption_time):
        """
        Handle detected interruption event.
        
        Args:
            transcription: The transcribed interruption text
            interruption_time: Time when interruption occurred
        """
        logging.info(f"Interruption detected at {interruption_time}: '{transcription[:50]}...'")
        
        # Record interruption with ADHD detector if available
        if self.adhd_detector:
            self.adhd_detector.record_question_interruption(
                interruption_time=interruption_time,
                response_content=transcription
            )
        
        # Call interruption callback if set
        if self.interruption_callback:
            try:
                self.interruption_callback(transcription, interruption_time)
            except Exception as e:
                logging.error(f"Error in interruption callback: {e}")

    def _handle_valid_response(self, transcription, response_time):
        """
        Handle valid response (not an interruption).
        
        Args:
            transcription: The transcribed response text
            response_time: Time when response was detected
        """
        logging.info(f"Valid response detected at {response_time}: '{transcription[:50]}...'")
        
        # Call response callback if set
        if self.response_callback:
            try:
                self.response_callback(transcription, response_time)
            except Exception as e:
                logging.error(f"Error in response callback: {e}")

    def get_question_monitoring_status(self):
        """
        Get current status of question monitoring.
        
        Returns:
            dict: Status information including timing and state
        """
        return {
            "monitoring_active": self.question_monitoring_active,
            "question_start_time": self.current_question_start_time,
            "question_completion_time": self.current_question_completion_time,
            "question_presentation_complete": self.current_question_completion_time is not None,
            "time_since_question_start": (
                time.time() - self.current_question_start_time 
                if self.current_question_start_time else None
            ),
            "time_since_question_completion": (
                time.time() - self.current_question_completion_time 
                if self.current_question_completion_time else None
            )
        }

    def set_interruption_detection_parameters(self, detection_delay=None, response_timeout=None):
        """
        Configure interruption detection parameters.
        
        Args:
            detection_delay: Minimum delay to consider as interruption (seconds)
            response_timeout: Maximum time to wait for response after question completion (seconds)
        """
        if detection_delay is not None:
            self.interruption_detection_delay = detection_delay
            logging.info(f"Interruption detection delay set to {detection_delay}s")
            
        if response_timeout is not None:
            self.response_timeout = response_timeout
            logging.info(f"Response timeout set to {response_timeout}s")

    def is_question_presentation_active(self):
        """
        Check if question presentation is currently active (started but not completed).
        
        Returns:
            bool: True if question is being presented, False otherwise
        """
        return (self.question_monitoring_active and 
                self.current_question_start_time is not None and 
                self.current_question_completion_time is None)

    def get_time_since_question_start(self):
        """
        Get time elapsed since current question started.
        
        Returns:
            float or None: Time in seconds, or None if no question active
        """
        if self.current_question_start_time:
            return time.time() - self.current_question_start_time
        return None

    def get_time_since_question_completion(self):
        """
        Get time elapsed since current question completed.
        
        Returns:
            float or None: Time in seconds, or None if question not completed
        """
        if self.current_question_completion_time:
            return time.time() - self.current_question_completion_time
        return None


# Test function for interruption detection
def test_interruption_detection():
    """Test the interruption detection functionality"""
    print("Testing interruption detection functionality...")
    
    # Create recognizer with ADHD detector
    from adhd_detection import ADHDDetector
    adhd_detector = ADHDDetector()
    recognizer = ContinuousSpeechRecognizer(adhd_detector=adhd_detector)
    
    # Test callbacks
    interruptions_detected = []
    responses_detected = []
    
    def interruption_callback(transcription, timestamp):
        interruptions_detected.append((transcription, timestamp))
        print(f"INTERRUPTION: '{transcription}' at {timestamp}")
    
    def response_callback(transcription, timestamp):
        responses_detected.append((transcription, timestamp))
        print(f"RESPONSE: '{transcription}' at {timestamp}")
    
    # Test question monitoring setup
    print("1. Testing question monitoring setup...")
    recognizer.start_question_monitoring(
        interruption_callback=interruption_callback,
        response_callback=response_callback
    )
    
    status = recognizer.get_question_monitoring_status()
    print(f"   Monitoring status: {status}")
    assert status["monitoring_active"] == True
    assert status["question_presentation_complete"] == False
    print("   ✓ Question monitoring setup successful")
    
    # Test interruption detection logic
    print("2. Testing interruption detection logic...")
    current_time = time.time()
    
    # Simulate speech during question presentation (should be interruption)
    is_interruption = recognizer._is_speech_interruption(current_time + 1.0)
    print(f"   Speech during presentation: {is_interruption}")
    assert is_interruption == True
    
    # Mark question as completed
    recognizer.mark_question_completion(current_time + 5.0)
    
    # Simulate speech after question completion (should be valid response)
    is_interruption = recognizer._is_speech_interruption(current_time + 6.0)
    print(f"   Speech after completion: {is_interruption}")
    assert is_interruption == False
    print("   ✓ Interruption detection logic working correctly")
    
    # Test parameter configuration
    print("3. Testing parameter configuration...")
    recognizer.set_interruption_detection_parameters(detection_delay=1.0, response_timeout=15)
    assert recognizer.interruption_detection_delay == 1.0
    assert recognizer.response_timeout == 15
    print("   ✓ Parameter configuration successful")
    
    # Test monitoring stop
    print("4. Testing monitoring stop...")
    recognizer.stop_question_monitoring()
    status = recognizer.get_question_monitoring_status()
    assert status["monitoring_active"] == False
    print("   ✓ Question monitoring stopped successfully")
    
    print("All interruption detection tests passed! ✓")

# For backward compatibility
if __name__ == "__main__":
    # Test interruption detection if run directly
    test_interruption_detection()
    
    # Original transcription test
    audio_filepath = "test_speech_to_text.mp3"
    try:
        print(f"\nTesting transcription with: {audio_filepath}")
        print(transcribe_with_groq(audio_filepath))
    except Exception as e:
        print(f"An error occurred: {e}")
