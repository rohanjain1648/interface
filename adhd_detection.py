import webrtcvad
import pyaudio
import numpy as np
import speechbrain
from speechbrain.inference import SpeakerRecognition  # Updated import
import threading
import time
import logging
from collections import deque

logging.basicConfig(level=logging.INFO)

class ADHDDetector:
    def __init__(self, sample_rate=16000, frame_duration=30, window_size=5, interruption_threshold=3):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration  # ms
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.window_size = window_size  # sliding window for last N interactions
        self.interruption_threshold = interruption_threshold  # interruptions in window to flag ADHD

        # VAD
        self.vad = webrtcvad.Vad(3)  # Aggressiveness 0-3

        # Speaker recognition - simplified for now, we'll use a basic approach
        # self.speaker_recognizer = SpeakerRecognition.from_hparams(
        #     source="speechbrain/spkrec-ecapa-voxceleb",
        #     savedir="tmp/spkrec-ecapa-voxceleb"
        # )
        self.speaker_recognizer = None  # Disable for now to avoid issues

        # State
        self.speakers = {}  # speaker_id: embedding
        self.current_speaker = None
        self.speech_segments = []  # list of (start_time, end_time, speaker_id)
        self.interruptions = deque(maxlen=window_size)
        self.adhd_flagged = False

        # Question-based detection state
        self.question_interruptions = []  # list of interruption events during questions
        self.question_assessment_active = False
        self.current_question_start_time = None
        self.question_based_adhd_flagged = False
        self.question_interruption_threshold = 3  # 3 or more interruptions flag ADHD

        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.thread = None

    def start_audio_stream(self):
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frame_size
        )
        self.running = True
        self.thread = threading.Thread(target=self._process_audio)
        self.thread.start()

    def stop_audio_stream(self):
        self.running = False
        if self.thread:
            self.thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def _process_audio(self):
        while self.running:
            try:
                data = self.stream.read(self.frame_size, exception_on_overflow=False)
                audio_frame = np.frombuffer(data, dtype=np.int16)
                is_speech = self.vad.is_speech(data, self.sample_rate)

                if is_speech:
                    speaker_id = self._identify_speaker(audio_frame)
                    self._update_speech_segments(speaker_id)
                    
                    # Check for question interruption if assessment is active
                    if (self.question_assessment_active and 
                        self.current_question_start_time is not None and
                        speaker_id != "system"):  # Assume system is presenting the question
                        self.record_question_interruption()
                else:
                    self.current_speaker = None
            except Exception as e:
                logging.error(f"Audio processing error: {e}")

    def _identify_speaker(self, audio_frame):
        # Simplified: just alternate between two speakers for demo
        # In real implementation, use speaker recognition
        if not self.speakers:
            self.speakers["speaker_0"] = None
            return "speaker_0"
        elif len(self.speakers) == 1:
            self.speakers["speaker_1"] = None
            return "speaker_1"
        else:
            # Alternate for demo
            return "speaker_0" if self.current_speaker == "speaker_1" else "speaker_1"

    def _update_speech_segments(self, speaker_id):
        current_time = time.time()
        if self.current_speaker != speaker_id:
            if self.current_speaker is not None:
                # End previous segment
                if self.speech_segments:
                    self.speech_segments[-1] = (self.speech_segments[-1][0], current_time, self.speech_segments[-1][2])
                # Check for interruption
                if len(self.speech_segments) >= 2:
                    prev_end, _, prev_speaker = self.speech_segments[-1]
                    if current_time - prev_end < 0.5 and prev_speaker != speaker_id:  # Overlap threshold
                        self.interruptions.append(1)
                        logging.info(f"Interruption detected: {speaker_id} interrupted {prev_speaker}")
                    else:
                        self.interruptions.append(0)
            # Start new segment
            self.speech_segments.append((current_time, None, speaker_id))
            self.current_speaker = speaker_id

    def check_adhd(self):
        """Check for ADHD indicators using audio-based detection"""
        if len(self.interruptions) >= self.window_size:
            interruption_count = sum(self.interruptions)
            if interruption_count >= self.interruption_threshold:
                self.adhd_flagged = True
                return True
        return False

    def check_combined_adhd(self):
        """Check for ADHD indicators using both audio-based and question-based detection"""
        audio_adhd = self.check_adhd()
        question_results = self.get_question_based_results()
        question_adhd = question_results["flagged"]
        
        return audio_adhd or question_adhd

    def get_adhd_status(self):
        return {
            "flagged": self.adhd_flagged,
            "interruptions_in_window": sum(self.interruptions),
            "window_size": self.window_size,
            "threshold": self.interruption_threshold
        }

    def start_question_assessment(self):
        """Initialize question-based ADHD assessment"""
        self.question_assessment_active = True
        self.question_interruptions = []
        self.question_based_adhd_flagged = False
        logging.info("Question-based ADHD assessment started")

    def start_question_presentation(self):
        """Mark the start of a question being presented"""
        self.current_question_start_time = time.time()
        logging.info("Question presentation started")

    def end_question_presentation(self):
        """Mark the end of a question being presented"""
        self.current_question_start_time = None
        logging.info("Question presentation ended")

    def record_question_interruption(self, interruption_time=None, response_content=""):
        """Record an interruption event during question presentation
        
        Args:
            interruption_time: Time when interruption occurred (defaults to current time)
            response_content: Content of the user's response/interruption
        """
        if not self.question_assessment_active:
            logging.warning("Attempted to record question interruption when assessment not active")
            return

        if interruption_time is None:
            interruption_time = time.time()

        interruption_event = {
            "timestamp": interruption_time,
            "question_start_time": self.current_question_start_time,
            "response_content": response_content,
            "is_during_presentation": self.current_question_start_time is not None
        }

        self.question_interruptions.append(interruption_event)
        logging.info(f"Question interruption recorded: {len(self.question_interruptions)} total interruptions")

    def get_question_based_results(self):
        """Get results from question-based ADHD detection
        
        Returns:
            dict: Results including interruption count, flagged status, and details
        """
        interruption_count = len([
            event for event in self.question_interruptions 
            if event["is_during_presentation"]
        ])
        
        self.question_based_adhd_flagged = interruption_count >= self.question_interruption_threshold

        return {
            "flagged": self.question_based_adhd_flagged,
            "total_interruptions": interruption_count,
            "threshold": self.question_interruption_threshold,
            "interruption_events": self.question_interruptions,
            "assessment_active": self.question_assessment_active
        }

    def combine_detection_results(self):
        """Combine audio-based and question-based ADHD detection results
        
        Returns:
            dict: Combined results with overall ADHD flagging status
        """
        audio_results = self.get_adhd_status()
        question_results = self.get_question_based_results()

        # ADHD is flagged if either detection method flags it
        overall_flagged = audio_results["flagged"] or question_results["flagged"]

        return {
            "overall_flagged": overall_flagged,
            "audio_based": audio_results,
            "question_based": question_results,
            "detection_methods_used": {
                "audio_based": True,
                "question_based": self.question_assessment_active
            }
        }

    def end_question_assessment(self):
        """End the question-based ADHD assessment"""
        self.question_assessment_active = False
        self.current_question_start_time = None
        logging.info("Question-based ADHD assessment ended")

# Example usage
if __name__ == "__main__":
    detector = ADHDDetector()
    detector.start_audio_stream()
    time.sleep(10)  # Run for 10 seconds
    detector.stop_audio_stream()
    print(detector.get_adhd_status())
