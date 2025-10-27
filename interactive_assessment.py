"""
Interactive ADHD Assessment Module

This module provides the InteractiveADHDAssessment class that orchestrates
a 5-question assessment to detect potential ADHD indicators based on user
interruption patterns during question presentation.
"""

import time
import logging
import threading
import json
import os
import pickle
from datetime import datetime
from typing import List, Dict, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from text_to_speech import text_to_speech_with_elevenlabs

logging.basicConfig(level=logging.INFO)

class AssessmentState(Enum):
    """Enumeration of possible assessment states"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    PAUSED = "paused"
    ERROR = "error"
    RECOVERING = "recovering"

class AssessmentError(Exception):
    """Custom exception for assessment-related errors"""
    pass

class AudioSystemError(AssessmentError):
    """Exception for audio system failures"""
    pass

class UserDisconnectionError(AssessmentError):
    """Exception for user disconnection scenarios"""
    pass

@dataclass
class QuestionResponse:
    """Data class to store question response information"""
    question_index: int
    question_text: str
    start_time: float
    completion_time: Optional[float]
    user_response: Optional[str]
    was_interrupted: bool
    interruption_time: Optional[float]
    presentation_duration: Optional[float] = None
    timeout_occurred: bool = False

class InteractiveADHDAssessment:
    """
    Main assessment orchestration class that manages the 5-question ADHD assessment.
    
    This class handles question storage, state management, assessment flow control,
    and result calculation based on user interruption patterns.
    """
    
    # Predefined ADHD assessment questions focusing on attention and impulse control
    ASSESSMENT_QUESTIONS = [
        "Tell me about your typical morning routine. How do you usually start your day and what helps you stay organized?",
        "Describe a time when you had to focus on a task for an extended period. What strategies do you use to maintain concentration?",
        "How do you typically handle interruptions when you're trying to complete something important?",
        "Walk me through how you organize and prioritize your daily tasks or responsibilities.",
        "Think about a recent situation where you had to wait for something. How did you manage that waiting period?"
    ]
    
    def __init__(self, 
                 interruption_threshold: int = 3,
                 question_timeout: int = 10,
                 audio_output_path: str = "assessment_audio.wav",
                 results_storage_path: str = "assessment_results",
                 session_storage_path: str = "assessment_sessions",
                 auto_save_interval: int = 30,
                 max_retry_attempts: int = 3,
                 recovery_timeout: int = 60):
        """
        Initialize the InteractiveADHDAssessment.
        
        Args:
            interruption_threshold: Number of interruptions needed to flag ADHD (default: 3)
            question_timeout: Seconds to wait for response after question completion (default: 10)
            audio_output_path: Path for audio output files
            results_storage_path: Directory path for storing assessment results
            session_storage_path: Directory path for storing session state
            auto_save_interval: Seconds between automatic session saves (default: 30)
            max_retry_attempts: Maximum retry attempts for failed operations (default: 3)
            recovery_timeout: Timeout for recovery operations in seconds (default: 60)
        """
        self.interruption_threshold = interruption_threshold
        self.question_timeout = question_timeout
        self.audio_output_path = audio_output_path
        self.results_storage_path = results_storage_path
        self.session_storage_path = session_storage_path
        self.auto_save_interval = auto_save_interval
        self.max_retry_attempts = max_retry_attempts
        self.recovery_timeout = recovery_timeout
        
        # Create storage directories if they don't exist
        os.makedirs(self.results_storage_path, exist_ok=True)
        os.makedirs(self.session_storage_path, exist_ok=True)
        
        # Assessment state
        self.state = AssessmentState.NOT_STARTED
        self.current_question_index = 0
        self.interruption_count = 0
        self.responses: List[QuestionResponse] = []
        
        # Timing and flow control
        self.assessment_start_time: Optional[float] = None
        self.current_question_start_time: Optional[float] = None
        self.current_question_completion_time: Optional[float] = None
        self.is_presenting_question = False
        self.question_timeout_timer: Optional[threading.Timer] = None
        self.presentation_thread: Optional[threading.Thread] = None
        
        # Result storage
        self.current_assessment_id: Optional[str] = None
        self.stored_results: Optional[Dict] = None
        
        # Session state management
        self.session_id: Optional[str] = None
        self.last_save_time: Optional[float] = None
        self.auto_save_timer: Optional[threading.Timer] = None
        self.session_file_path: Optional[str] = None
        
        # Error handling and recovery
        self.error_count = 0
        self.last_error_time: Optional[float] = None
        self.recovery_attempts = 0
        self.is_recovering = False
        self.error_log: List[Dict] = []
        
        # Callbacks for external integration
        self.on_question_start_callback: Optional[Callable] = None
        self.on_question_complete_callback: Optional[Callable] = None
        self.on_interruption_callback: Optional[Callable] = None
        self.on_assessment_complete_callback: Optional[Callable] = None
        self.on_error_callback: Optional[Callable] = None
        self.on_recovery_callback: Optional[Callable] = None
        
        logging.info("InteractiveADHDAssessment initialized with state management and error handling")
    
    def start_assessment(self) -> bool:
        """
        Start the ADHD assessment process with error handling and state management.
        
        Returns:
            bool: True if assessment started successfully, False otherwise
        """
        if self.state not in [AssessmentState.NOT_STARTED, AssessmentState.RECOVERING]:
            logging.warning(f"Cannot start assessment. Current state: {self.state}")
            return False
        
        try:
            # Check for existing session to recover
            if self.state == AssessmentState.NOT_STARTED:
                latest_session = self._find_latest_session_file()
                if latest_session:
                    logging.info("Found existing session - attempting to load")
                    if self.load_session_state():
                        if self.state == AssessmentState.RECOVERING:
                            logging.info("Continuing from recovered session")
                            return self.continue_assessment()
            
            # Start fresh assessment
            self.state = AssessmentState.IN_PROGRESS
            self.assessment_start_time = time.time()
            self.current_question_index = 0
            self.interruption_count = 0
            self.responses.clear()
            self.error_count = 0
            self.recovery_attempts = 0
            
            # Generate session ID and start auto-save
            self.session_id = f"assessment_{int(time.time())}_{os.getpid()}"
            self.start_auto_save()
            
            # Save initial state
            self.save_session_state()
            
            logging.info("ADHD assessment started with state management")
            
            # Present the first question
            return self.present_current_question()
            
        except Exception as e:
            logging.error(f"Error starting assessment: {e}")
            self._handle_error(e, "start_assessment")
            return False
    
    def present_current_question(self) -> bool:
        """
        Present the current question using text-to-speech with timing tracking.
        
        Returns:
            bool: True if question presented successfully, False otherwise
        """
        if self.current_question_index >= len(self.ASSESSMENT_QUESTIONS):
            logging.warning("No more questions to present")
            return False
        
        try:
            question_text = self.ASSESSMENT_QUESTIONS[self.current_question_index]
            self.current_question_start_time = time.time()
            self.is_presenting_question = True
            
            # Create response object for this question
            response = QuestionResponse(
                question_index=self.current_question_index,
                question_text=question_text,
                start_time=self.current_question_start_time,
                completion_time=None,
                user_response=None,
                was_interrupted=False,
                interruption_time=None
            )
            self.responses.append(response)
            
            logging.info(f"Presenting question {self.current_question_index + 1}: {question_text[:50]}...")
            
            # Trigger callback if set
            if self.on_question_start_callback:
                self.on_question_start_callback(self.current_question_index, question_text)
            
            # Start question presentation in a separate thread to track timing
            self.presentation_thread = threading.Thread(
                target=self._present_question_with_timing,
                args=(question_text,)
            )
            self.presentation_thread.start()
            
            return True
            
        except Exception as e:
            logging.error(f"Error presenting question: {e}")
            self.is_presenting_question = False
            return False
    
    def _present_question_with_timing(self, question_text: str) -> None:
        """
        Present question with precise timing tracking and error handling.
        
        Args:
            question_text: The question text to present
        """
        try:
            # Record presentation start time
            presentation_start = time.time()
            
            # Use text-to-speech to present the question with error handling
            try:
                text_to_speech_with_elevenlabs(question_text, self.audio_output_path)
            except Exception as audio_error:
                # Handle audio system errors
                logging.warning(f"Audio system error during question presentation: {audio_error}")
                if not self._handle_error(AudioSystemError(f"TTS failed: {audio_error}"), "present_question_audio"):
                    # If audio recovery fails, continue without audio
                    logging.warning("Continuing assessment without audio due to system failure")
            
            # Record presentation completion time
            presentation_end = time.time()
            presentation_duration = presentation_end - presentation_start
            
            # Update response record with timing information
            if self.responses and self.current_question_index < len(self.responses):
                self.responses[self.current_question_index].completion_time = presentation_end
                self.responses[self.current_question_index].presentation_duration = presentation_duration
            
            # Mark question presentation as complete
            self.is_presenting_question = False
            self.current_question_completion_time = presentation_end
            
            # Save state after question presentation
            self.save_session_state()
            
            logging.info(f"Question {self.current_question_index + 1} presentation completed in {presentation_duration:.2f}s")
            
            # Trigger callback if set
            if self.on_question_complete_callback:
                self.on_question_complete_callback(self.current_question_index)
            
            # Start timeout timer for user response
            self._start_response_timeout()
            
        except Exception as e:
            logging.error(f"Error in question presentation thread: {e}")
            self.is_presenting_question = False
            self._handle_error(e, "present_question_timing")
    
    def _start_response_timeout(self) -> None:
        """
        Start timeout timer for user response after question completion.
        """
        if self.question_timeout_timer:
            self.question_timeout_timer.cancel()
        
        self.question_timeout_timer = threading.Timer(
            self.question_timeout,
            self._handle_response_timeout
        )
        self.question_timeout_timer.start()
        
        logging.info(f"Response timeout timer started for {self.question_timeout} seconds")
    
    def _handle_response_timeout(self) -> None:
        """
        Handle timeout when user doesn't respond within the specified time.
        """
        try:
            logging.info(f"Response timeout occurred for question {self.current_question_index + 1}")
            
            # Mark timeout in response record
            if self.responses and self.current_question_index < len(self.responses):
                self.responses[self.current_question_index].timeout_occurred = True
                self.responses[self.current_question_index].user_response = "[No response - timeout]"
            
            # Proceed to next question
            self.proceed_to_next_question()
            
        except Exception as e:
            logging.error(f"Error handling response timeout: {e}")
    
    def _cancel_response_timeout(self) -> None:
        """
        Cancel the response timeout timer.
        """
        if self.question_timeout_timer:
            self.question_timeout_timer.cancel()
            self.question_timeout_timer = None
            logging.debug("Response timeout timer cancelled")
    
    def handle_user_response(self, response_text: str, response_time: Optional[float] = None) -> bool:
        """
        Handle user response to the current question with improved timing detection.
        
        Args:
            response_text: The user's response text
            response_time: Timestamp when response was detected (defaults to current time)
            
        Returns:
            bool: True if response handled successfully, False otherwise
        """
        if not response_time:
            response_time = time.time()
        
        try:
            # Cancel response timeout since user responded
            self._cancel_response_timeout()
            
            # Check if this is an interruption (user spoke during question presentation)
            if self.is_presenting_question and self.current_question_start_time:
                self.record_interruption(response_time)
                logging.info(f"Interruption detected during question presentation")
            
            # Store the user's response
            if self.responses and self.current_question_index < len(self.responses):
                self.responses[self.current_question_index].user_response = response_text
            
            logging.info(f"User response recorded for question {self.current_question_index + 1}: '{response_text[:50]}...'")
            
            # Move to next question or complete assessment
            return self.proceed_to_next_question()
            
        except Exception as e:
            logging.error(f"Error handling user response: {e}")
            return False
    
    def record_interruption(self, interruption_time: float) -> None:
        """
        Record an interruption event during question presentation.
        
        Args:
            interruption_time: Timestamp when interruption occurred
        """
        self.interruption_count += 1
        
        # Update the current response record
        if self.responses and self.current_question_index < len(self.responses):
            self.responses[self.current_question_index].was_interrupted = True
            self.responses[self.current_question_index].interruption_time = interruption_time
        
        logging.info(f"Interruption recorded. Total interruptions: {self.interruption_count}")
        
        # Trigger callback if set
        if self.on_interruption_callback:
            self.on_interruption_callback(self.interruption_count, interruption_time)
    
    def proceed_to_next_question(self) -> bool:
        """
        Proceed to the next question or complete the assessment.
        
        Returns:
            bool: True if successfully proceeded, False otherwise
        """
        try:
            self.current_question_index += 1
            
            # Check if assessment is complete
            if self.current_question_index >= len(self.ASSESSMENT_QUESTIONS):
                return self.complete_assessment()
            
            # Present next question
            return self.present_current_question()
            
        except Exception as e:
            logging.error(f"Error proceeding to next question: {e}")
            return False
    
    def complete_assessment(self) -> bool:
        """
        Complete the assessment, calculate results, and store them with error handling.
        
        Returns:
            bool: True if assessment completed successfully, False otherwise
        """
        try:
            self.state = AssessmentState.COMPLETED
            
            # Stop auto-save timer
            self._stop_auto_save()
            
            # Calculate results
            results = self.calculate_results()
            
            # Store results automatically with retry logic
            storage_success = False
            for attempt in range(self.max_retry_attempts):
                try:
                    storage_success = self.store_results(results)
                    if storage_success:
                        break
                except Exception as storage_error:
                    logging.warning(f"Result storage attempt {attempt + 1} failed: {storage_error}")
                    if attempt < self.max_retry_attempts - 1:
                        time.sleep(1)  # Brief delay before retry
            
            if not storage_success:
                logging.warning("Failed to store assessment results after all attempts, but assessment completed")
            
            # Save final session state
            self.save_session_state()
            
            logging.info(f"Assessment completed. ADHD flagged: {results['adhd_flagged']}, "
                        f"Interruptions: {results['interruption_count']}/{results['interruption_threshold']}")
            
            # Trigger callback if set
            if self.on_assessment_complete_callback:
                self.on_assessment_complete_callback(results)
            
            # Clean up session file after successful completion
            self.delete_session_state()
            
            return True
            
        except Exception as e:
            logging.error(f"Error completing assessment: {e}")
            self._handle_error(e, "complete_assessment")
            return False
    
    def calculate_results(self) -> Dict:
        """
        Calculate comprehensive assessment results based on interruption patterns.
        
        Returns:
            Dict: Assessment results including ADHD flagging, statistics, and detailed analysis
        """
        total_questions = len(self.ASSESSMENT_QUESTIONS)
        completed_questions = len(self.responses)
        
        # Core ADHD flagging logic - Requirements 1.5, 3.1, 3.2
        adhd_flagged = self.interruption_count >= self.interruption_threshold
        
        # Calculate timing statistics
        total_assessment_time = 0
        if self.assessment_start_time and self.responses:
            last_response_time = max(
                (r.completion_time or r.start_time for r in self.responses if r.completion_time or r.start_time),
                default=time.time()
            )
            total_assessment_time = last_response_time - self.assessment_start_time
        
        # Enhanced result calculation with detailed analysis
        results = {
            # Core results - Requirements 3.1, 3.2, 3.3
            "adhd_flagged": adhd_flagged,
            "interruption_count": self.interruption_count,
            "interruption_threshold": self.interruption_threshold,
            "flag_threshold_met": self.interruption_count >= self.interruption_threshold,
            
            # Assessment completion metrics
            "total_questions": total_questions,
            "completed_questions": completed_questions,
            "completion_rate": completed_questions / total_questions if total_questions > 0 else 0,
            "assessment_completed": completed_questions == total_questions,
            
            # Timing analysis
            "total_assessment_time": total_assessment_time,
            "average_time_per_question": total_assessment_time / completed_questions if completed_questions > 0 else 0,
            
            # Response data
            "responses": self.responses,
            "response_summary": self._generate_response_summary(),
            
            # Result messages - Requirements 3.1, 3.2, 3.4
            "assessment_message": self._generate_assessment_message(adhd_flagged),
            "detailed_message": self._generate_detailed_message(adhd_flagged),
            "disclaimer": self._generate_disclaimer(),
            
            # Timing and behavioral statistics
            "timing_statistics": self._calculate_timing_statistics(),
            "behavioral_analysis": self._calculate_behavioral_analysis(),
            
            # Metadata
            "assessment_timestamp": time.time(),
            "assessment_id": f"assessment_{int(time.time())}",
            "assessment_version": "1.0"
        }
        
        return results
    
    def _calculate_timing_statistics(self) -> Dict:
        """
        Calculate detailed timing statistics for the assessment.
        
        Returns:
            Dict: Timing statistics including averages and patterns
        """
        if not self.responses:
            return {}
        
        presentation_durations = [r.presentation_duration for r in self.responses if r.presentation_duration]
        interruption_times = []
        response_delays = []
        
        for response in self.responses:
            if response.was_interrupted and response.interruption_time and response.start_time:
                # Time from question start to interruption
                interruption_delay = response.interruption_time - response.start_time
                interruption_times.append(interruption_delay)
            
            if (response.completion_time and response.start_time and 
                response.user_response and not response.timeout_occurred):
                # Time from question completion to response (if response came after completion)
                if not response.was_interrupted:
                    response_delay = response.completion_time - response.start_time
                    response_delays.append(response_delay)
        
        stats = {
            "total_questions_presented": len(self.responses),
            "questions_with_interruptions": sum(1 for r in self.responses if r.was_interrupted),
            "questions_with_timeouts": sum(1 for r in self.responses if r.timeout_occurred),
            "average_presentation_duration": sum(presentation_durations) / len(presentation_durations) if presentation_durations else 0,
            "total_presentation_time": sum(presentation_durations) if presentation_durations else 0,
            "average_interruption_delay": sum(interruption_times) / len(interruption_times) if interruption_times else 0,
            "fastest_interruption": min(interruption_times) if interruption_times else None,
            "slowest_interruption": max(interruption_times) if interruption_times else None,
            "interruption_rate": len(interruption_times) / len(self.responses) if self.responses else 0
        }
        
        return stats
    
    def _calculate_behavioral_analysis(self) -> Dict:
        """
        Calculate detailed behavioral analysis from assessment responses.
        
        Returns:
            Dict: Behavioral patterns and analysis
        """
        if not self.responses:
            return {}
        
        # Interruption timing analysis
        interruption_delays = []
        early_interruptions = 0  # Interruptions within first 2 seconds
        mid_interruptions = 0    # Interruptions between 2-5 seconds
        late_interruptions = 0   # Interruptions after 5 seconds
        
        for response in self.responses:
            if response.was_interrupted and response.interruption_time and response.start_time:
                delay = response.interruption_time - response.start_time
                interruption_delays.append(delay)
                
                if delay <= 2.0:
                    early_interruptions += 1
                elif delay <= 5.0:
                    mid_interruptions += 1
                else:
                    late_interruptions += 1
        
        # Response pattern analysis
        consecutive_interruptions = 0
        max_consecutive = 0
        current_streak = 0
        
        for response in self.responses:
            if response.was_interrupted:
                current_streak += 1
                max_consecutive = max(max_consecutive, current_streak)
            else:
                current_streak = 0
        
        # Question-specific analysis
        question_analysis = []
        for i, response in enumerate(self.responses):
            question_data = {
                "question_number": i + 1,
                "was_interrupted": response.was_interrupted,
                "timeout_occurred": response.timeout_occurred,
                "presentation_duration": response.presentation_duration,
                "interruption_delay": (response.interruption_time - response.start_time) if response.was_interrupted and response.interruption_time and response.start_time else None,
                "response_provided": bool(response.user_response and response.user_response not in ["[No response - timeout]", "[Manually skipped]"])
            }
            question_analysis.append(question_data)
        
        return {
            "interruption_timing": {
                "early_interruptions": early_interruptions,
                "mid_interruptions": mid_interruptions,
                "late_interruptions": late_interruptions,
                "average_interruption_delay": sum(interruption_delays) / len(interruption_delays) if interruption_delays else 0,
                "fastest_interruption": min(interruption_delays) if interruption_delays else None,
                "slowest_interruption": max(interruption_delays) if interruption_delays else None
            },
            "interruption_patterns": {
                "max_consecutive_interruptions": max_consecutive,
                "interruption_consistency": len(interruption_delays) / len(self.responses) if self.responses else 0,
                "impulsivity_score": early_interruptions / len(self.responses) if self.responses else 0
            },
            "question_analysis": question_analysis,
            "behavioral_indicators": {
                "high_impulsivity": early_interruptions >= 2,
                "consistent_interruption": len(interruption_delays) >= 3,
                "attention_difficulty": sum(1 for r in self.responses if r.timeout_occurred) >= 2
            }
        }
    
    def store_results(self, results: Dict) -> bool:
        """
        Store assessment results to file system for later retrieval.
        
        Args:
            results: Assessment results dictionary to store
            
        Returns:
            bool: True if results stored successfully, False otherwise
        """
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            assessment_id = results.get("assessment_id", f"assessment_{timestamp}")
            filename = f"{assessment_id}.json"
            filepath = os.path.join(self.results_storage_path, filename)
            
            # Prepare results for JSON serialization
            serializable_results = self._prepare_results_for_storage(results)
            
            # Store results to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
            # Update current stored results
            self.stored_results = serializable_results
            self.current_assessment_id = assessment_id
            
            logging.info(f"Assessment results stored to {filepath}")
            return True
            
        except Exception as e:
            logging.error(f"Error storing assessment results: {e}")
            return False
    
    def _prepare_results_for_storage(self, results: Dict) -> Dict:
        """
        Prepare results dictionary for JSON serialization.
        
        Args:
            results: Raw results dictionary
            
        Returns:
            Dict: Serializable results dictionary
        """
        serializable_results = results.copy()
        
        # Convert QuestionResponse objects to dictionaries
        if "responses" in serializable_results:
            serializable_results["responses"] = [
                asdict(response) for response in serializable_results["responses"]
            ]
        
        # Add human-readable timestamp
        if "assessment_timestamp" in serializable_results:
            serializable_results["assessment_datetime"] = datetime.fromtimestamp(
                serializable_results["assessment_timestamp"]
            ).isoformat()
        
        return serializable_results
    
    def retrieve_results(self, assessment_id: Optional[str] = None) -> Optional[Dict]:
        """
        Retrieve stored assessment results.
        
        Args:
            assessment_id: Specific assessment ID to retrieve (defaults to current assessment)
            
        Returns:
            Optional[Dict]: Retrieved results or None if not found
        """
        try:
            if assessment_id is None:
                assessment_id = self.current_assessment_id
            
            if assessment_id is None:
                logging.warning("No assessment ID provided for retrieval")
                return None
            
            filename = f"{assessment_id}.json"
            filepath = os.path.join(self.results_storage_path, filename)
            
            if not os.path.exists(filepath):
                logging.warning(f"Assessment results file not found: {filepath}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            logging.info(f"Assessment results retrieved from {filepath}")
            return results
            
        except Exception as e:
            logging.error(f"Error retrieving assessment results: {e}")
            return None
    
    def list_stored_assessments(self) -> List[Dict]:
        """
        List all stored assessment results with summary information.
        
        Returns:
            List[Dict]: List of assessment summaries
        """
        try:
            assessments = []
            
            if not os.path.exists(self.results_storage_path):
                return assessments
            
            for filename in os.listdir(self.results_storage_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.results_storage_path, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            results = json.load(f)
                        
                        # Create summary
                        summary = {
                            "assessment_id": results.get("assessment_id", filename[:-5]),
                            "filename": filename,
                            "timestamp": results.get("assessment_timestamp"),
                            "datetime": results.get("assessment_datetime"),
                            "adhd_flagged": results.get("adhd_flagged", False),
                            "interruption_count": results.get("interruption_count", 0),
                            "completed_questions": results.get("completed_questions", 0),
                            "total_questions": results.get("total_questions", 0),
                            "assessment_completed": results.get("assessment_completed", False)
                        }
                        assessments.append(summary)
                        
                    except Exception as e:
                        logging.warning(f"Error reading assessment file {filename}: {e}")
                        continue
            
            # Sort by timestamp (newest first)
            assessments.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
            
            return assessments
            
        except Exception as e:
            logging.error(f"Error listing stored assessments: {e}")
            return []
    
    def get_latest_results(self) -> Optional[Dict]:
        """
        Get the most recent assessment results.
        
        Returns:
            Optional[Dict]: Latest assessment results or None if none found
        """
        assessments = self.list_stored_assessments()
        if assessments:
            latest = assessments[0]
            return self.retrieve_results(latest["assessment_id"])
        return None
    
    def delete_stored_results(self, assessment_id: str) -> bool:
        """
        Delete stored assessment results.
        
        Args:
            assessment_id: Assessment ID to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            filename = f"{assessment_id}.json"
            filepath = os.path.join(self.results_storage_path, filename)
            
            if os.path.exists(filepath):
                os.remove(filepath)
                logging.info(f"Assessment results deleted: {filepath}")
                
                # Clear current stored results if this was the current assessment
                if self.current_assessment_id == assessment_id:
                    self.stored_results = None
                    self.current_assessment_id = None
                
                return True
            else:
                logging.warning(f"Assessment results file not found for deletion: {filepath}")
                return False
                
        except Exception as e:
            logging.error(f"Error deleting assessment results: {e}")
            return False
    
    def export_results_summary(self, output_path: Optional[str] = None) -> Optional[str]:
        """
        Export a summary of all stored assessment results.
        
        Args:
            output_path: Path for the summary file (defaults to results directory)
            
        Returns:
            Optional[str]: Path to the exported summary file or None if failed
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(self.results_storage_path, f"assessment_summary_{timestamp}.json")
            
            assessments = self.list_stored_assessments()
            
            # Create comprehensive summary
            summary = {
                "export_timestamp": time.time(),
                "export_datetime": datetime.now().isoformat(),
                "total_assessments": len(assessments),
                "assessments_with_adhd_flags": sum(1 for a in assessments if a.get("adhd_flagged", False)),
                "average_interruption_count": sum(a.get("interruption_count", 0) for a in assessments) / len(assessments) if assessments else 0,
                "completion_rate": sum(1 for a in assessments if a.get("assessment_completed", False)) / len(assessments) if assessments else 0,
                "assessments": assessments
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Assessment summary exported to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error exporting assessment summary: {e}")
            return None
    
    def format_results_for_display(self, results: Optional[Dict] = None) -> str:
        """
        Format assessment results for user-friendly display.
        
        Args:
            results: Results dictionary (defaults to current stored results)
            
        Returns:
            str: Formatted results text for display
        """
        if results is None:
            results = self.stored_results or self.calculate_results()
        
        if not results:
            return "No assessment results available."
        
        # Header
        display_lines = []
        display_lines.append("=" * 60)
        display_lines.append("🧠 ADHD ASSESSMENT RESULTS")
        display_lines.append("=" * 60)
        
        # Assessment completion status
        if results.get("assessment_completed", False):
            display_lines.append("✅ Assessment Status: COMPLETED")
        else:
            completed = results.get("completed_questions", 0)
            total = results.get("total_questions", 5)
            display_lines.append(f"⚠️ Assessment Status: PARTIALLY COMPLETED ({completed}/{total} questions)")
        
        display_lines.append("")
        
        # Main result
        if results.get("adhd_flagged", False):
            display_lines.append("🔴 RESULT: ADHD INDICATORS DETECTED")
            display_lines.append(f"   Interruptions: {results.get('interruption_count', 0)} (≥{results.get('interruption_threshold', 3)} threshold)")
        else:
            display_lines.append("🟢 RESULT: NO ADHD INDICATORS DETECTED")
            display_lines.append(f"   Interruptions: {results.get('interruption_count', 0)} (<{results.get('interruption_threshold', 3)} threshold)")
        
        display_lines.append("")
        
        # Detailed analysis
        display_lines.append("📊 DETAILED ANALYSIS:")
        
        # Response summary
        response_summary = results.get("response_summary", {})
        if response_summary:
            display_lines.append(f"   • Total Questions: {response_summary.get('total_responses', 0)}")
            display_lines.append(f"   • Interrupted Questions: {response_summary.get('interrupted_responses', 0)}")
            display_lines.append(f"   • Timeout Questions: {response_summary.get('timeout_responses', 0)}")
            display_lines.append(f"   • Valid Responses: {response_summary.get('valid_responses', 0)}")
            
            interruption_rate = response_summary.get('interruption_rate', 0) * 100
            display_lines.append(f"   • Interruption Rate: {interruption_rate:.1f}%")
        
        # Timing statistics
        timing_stats = results.get("timing_statistics", {})
        if timing_stats:
            display_lines.append("")
            display_lines.append("⏱️ TIMING ANALYSIS:")
            
            total_time = results.get("total_assessment_time", 0)
            if total_time > 0:
                minutes = int(total_time // 60)
                seconds = int(total_time % 60)
                display_lines.append(f"   • Total Assessment Time: {minutes}m {seconds}s")
            
            avg_interruption = timing_stats.get("average_interruption_delay", 0)
            if avg_interruption > 0:
                display_lines.append(f"   • Average Interruption Delay: {avg_interruption:.1f}s")
            
            fastest = timing_stats.get("fastest_interruption")
            if fastest is not None:
                display_lines.append(f"   • Fastest Interruption: {fastest:.1f}s")
        
        # Behavioral analysis
        behavioral = results.get("behavioral_analysis", {})
        if behavioral:
            indicators = behavioral.get("behavioral_indicators", {})
            if indicators:
                display_lines.append("")
                display_lines.append("🧩 BEHAVIORAL INDICATORS:")
                
                if indicators.get("high_impulsivity", False):
                    display_lines.append("   • ⚡ High Impulsivity: Early interruptions detected")
                
                if indicators.get("consistent_interruption", False):
                    display_lines.append("   • 🔄 Consistent Interruption Pattern: Multiple interruptions across questions")
                
                if indicators.get("attention_difficulty", False):
                    display_lines.append("   • 🎯 Attention Difficulty: Multiple timeouts detected")
        
        # Disclaimer
        display_lines.append("")
        display_lines.append("⚠️ IMPORTANT DISCLAIMER:")
        display_lines.append(results.get("disclaimer", "This assessment is not diagnostic."))
        
        # Assessment metadata
        if results.get("assessment_datetime"):
            display_lines.append("")
            display_lines.append(f"📅 Assessment Date: {results['assessment_datetime']}")
        
        if results.get("assessment_id"):
            display_lines.append(f"🆔 Assessment ID: {results['assessment_id']}")
        
        display_lines.append("=" * 60)
        
        return "\n".join(display_lines)
    
    def format_results_summary(self, results: Optional[Dict] = None) -> str:
        """
        Format a brief summary of assessment results.
        
        Args:
            results: Results dictionary (defaults to current stored results)
            
        Returns:
            str: Brief formatted results summary
        """
        if results is None:
            results = self.stored_results or self.calculate_results()
        
        if not results:
            return "No assessment results available."
        
        # Brief summary format
        status = "COMPLETED" if results.get("assessment_completed", False) else "PARTIAL"
        flagged = "DETECTED" if results.get("adhd_flagged", False) else "NOT DETECTED"
        interruptions = results.get("interruption_count", 0)
        threshold = results.get("interruption_threshold", 3)
        
        return (f"Assessment {status} | ADHD Indicators: {flagged} | "
                f"Interruptions: {interruptions}/{threshold}")
    
    def get_results_for_ui_display(self, results: Optional[Dict] = None) -> Dict:
        """
        Get results formatted specifically for UI display components.
        
        Args:
            results: Results dictionary (defaults to current stored results)
            
        Returns:
            Dict: UI-formatted results data
        """
        if results is None:
            results = self.stored_results or self.calculate_results()
        
        if not results:
            return {"error": "No assessment results available"}
        
        # Format for UI components
        ui_results = {
            "status": "completed" if results.get("assessment_completed", False) else "partial",
            "adhd_detected": results.get("adhd_flagged", False),
            "interruption_count": results.get("interruption_count", 0),
            "interruption_threshold": results.get("interruption_threshold", 3),
            "completion_percentage": results.get("completion_rate", 0) * 100,
            "main_message": results.get("assessment_message", ""),
            "detailed_message": results.get("detailed_message", ""),
            "disclaimer": results.get("disclaimer", ""),
            "summary_text": self.format_results_summary(results),
            "full_display": self.format_results_for_display(results),
            "timestamp": results.get("assessment_datetime", ""),
            "assessment_id": results.get("assessment_id", "")
        }
        
        # Add visual indicators
        if ui_results["adhd_detected"]:
            ui_results["status_icon"] = "🔴"
            ui_results["status_color"] = "red"
            ui_results["result_class"] = "adhd-detected"
        else:
            ui_results["status_icon"] = "🟢"
            ui_results["status_color"] = "green"
            ui_results["result_class"] = "adhd-not-detected"
        
        # Add progress indicators
        ui_results["progress_bar_value"] = ui_results["completion_percentage"]
        ui_results["interruption_bar_value"] = min(100, (ui_results["interruption_count"] / ui_results["interruption_threshold"]) * 100)
        
        return ui_results
    
    def _generate_assessment_message(self, adhd_flagged: bool) -> str:
        """
        Generate appropriate assessment result message - Requirements 3.1, 3.2.
        
        Args:
            adhd_flagged: Whether ADHD indicators were detected
            
        Returns:
            str: Assessment result message
        """
        if adhd_flagged:
            return (
                f"ADHD indicators detected - {self.interruption_count} interruptions recorded "
                f"(threshold: {self.interruption_threshold}). Consider consulting a healthcare professional "
                "for a comprehensive evaluation."
            )
        else:
            return (
                f"No ADHD indicators detected - {self.interruption_count} interruptions recorded "
                f"(threshold: {self.interruption_threshold})."
            )
    
    def _generate_detailed_message(self, adhd_flagged: bool) -> str:
        """
        Generate detailed assessment result message with analysis.
        
        Args:
            adhd_flagged: Whether ADHD indicators were detected
            
        Returns:
            str: Detailed assessment result message
        """
        completed_questions = len(self.responses)
        total_questions = len(self.ASSESSMENT_QUESTIONS)
        
        message_parts = []
        
        # Assessment completion status
        if completed_questions == total_questions:
            message_parts.append(f"✅ Assessment completed successfully ({completed_questions}/{total_questions} questions)")
        else:
            message_parts.append(f"⚠️ Assessment partially completed ({completed_questions}/{total_questions} questions)")
        
        # Interruption analysis
        if adhd_flagged:
            message_parts.append(f"🔴 ADHD indicators detected: {self.interruption_count} interruptions (≥{self.interruption_threshold} threshold)")
            message_parts.append("📋 Recommendation: Consider professional consultation for comprehensive evaluation")
        else:
            message_parts.append(f"🟢 No ADHD indicators detected: {self.interruption_count} interruptions (<{self.interruption_threshold} threshold)")
        
        # Behavioral patterns
        interrupted_questions = sum(1 for r in self.responses if r.was_interrupted)
        if interrupted_questions > 0:
            message_parts.append(f"📊 Interruption pattern: {interrupted_questions}/{completed_questions} questions interrupted")
        
        return "\n".join(message_parts)
    
    def _generate_disclaimer(self) -> str:
        """
        Generate appropriate medical disclaimer - Requirements 3.4.
        
        Returns:
            str: Medical disclaimer text
        """
        return (
            "⚠️ IMPORTANT DISCLAIMER: This assessment is not diagnostic and should not replace "
            "professional medical advice. ADHD diagnosis requires comprehensive evaluation by a "
            "qualified healthcare professional. If you have concerns about ADHD symptoms, "
            "please consult with a medical professional."
        )
    
    def _generate_response_summary(self) -> Dict:
        """
        Generate summary of user responses during assessment.
        
        Returns:
            Dict: Summary of response patterns and behaviors
        """
        if not self.responses:
            return {}
        
        interrupted_responses = [r for r in self.responses if r.was_interrupted]
        timeout_responses = [r for r in self.responses if r.timeout_occurred]
        valid_responses = [r for r in self.responses if r.user_response and not r.timeout_occurred]
        
        return {
            "total_responses": len(self.responses),
            "interrupted_responses": len(interrupted_responses),
            "timeout_responses": len(timeout_responses),
            "valid_responses": len(valid_responses),
            "interruption_rate": len(interrupted_responses) / len(self.responses) if self.responses else 0,
            "timeout_rate": len(timeout_responses) / len(self.responses) if self.responses else 0,
            "response_rate": len(valid_responses) / len(self.responses) if self.responses else 0
        }
    
    def get_current_state(self) -> Dict:
        """
        Get current assessment state and progress.
        
        Returns:
            Dict: Current state information
        """
        return {
            "state": self.state.value,
            "current_question_index": self.current_question_index,
            "total_questions": len(self.ASSESSMENT_QUESTIONS),
            "interruption_count": self.interruption_count,
            "is_presenting_question": self.is_presenting_question,
            "progress_percentage": (self.current_question_index / len(self.ASSESSMENT_QUESTIONS)) * 100
        }
    
    def pause_assessment(self) -> bool:
        """
        Pause the current assessment.
        
        Returns:
            bool: True if successfully paused, False otherwise
        """
        if self.state == AssessmentState.IN_PROGRESS:
            self.state = AssessmentState.PAUSED
            self.is_presenting_question = False
            logging.info("Assessment paused")
            return True
        return False
    
    def resume_assessment(self) -> bool:
        """
        Resume a paused assessment.
        
        Returns:
            bool: True if successfully resumed, False otherwise
        """
        if self.state == AssessmentState.PAUSED:
            self.state = AssessmentState.IN_PROGRESS
            logging.info("Assessment resumed")
            return True
        return False
    
    def reset_assessment(self) -> None:
        """Reset the assessment to initial state with cleanup."""
        try:
            # Cancel any active timers
            self._cancel_response_timeout()
            self._stop_auto_save()
            
            # Wait for presentation thread to complete
            if self.presentation_thread and self.presentation_thread.is_alive():
                self.presentation_thread.join(timeout=2.0)
            
            # Clean up session state
            self.delete_session_state()
            
            # Reset all state variables
            self.state = AssessmentState.NOT_STARTED
            self.current_question_index = 0
            self.interruption_count = 0
            self.responses.clear()
            self.assessment_start_time = None
            self.current_question_start_time = None
            self.current_question_completion_time = None
            self.is_presenting_question = False
            self.question_timeout_timer = None
            self.presentation_thread = None
            
            # Reset error handling state
            self.error_count = 0
            self.recovery_attempts = 0
            self.is_recovering = False
            self.last_error_time = None
            self.error_log.clear()
            
            # Reset session management state
            self.session_id = None
            self.last_save_time = None
            self.session_file_path = None
            
            logging.info("Assessment reset with full cleanup")
            
        except Exception as e:
            logging.error(f"Error during assessment reset: {e}")
    
    def continue_assessment(self) -> bool:
        """
        Continue assessment from recovered state.
        
        Returns:
            bool: True if successfully continued, False otherwise
        """
        try:
            if self.state != AssessmentState.RECOVERING:
                logging.warning("Cannot continue assessment - not in recovering state")
                return False
            
            # Validate recovered state
            if self.current_question_index >= len(self.ASSESSMENT_QUESTIONS):
                logging.info("Assessment was already complete - marking as completed")
                return self.complete_assessment()
            
            # Resume assessment
            self.state = AssessmentState.IN_PROGRESS
            self.is_recovering = False
            
            # Restart auto-save
            self.start_auto_save()
            
            logging.info(f"Assessment continued from question {self.current_question_index + 1}")
            
            # Continue with current question or present next one
            if self.is_presenting_question:
                # Question was being presented when disconnection occurred
                logging.info("Resuming question presentation")
                return self.present_current_question()
            else:
                # Question was completed, waiting for response
                logging.info("Resuming from response waiting state")
                self._start_response_timeout()
                return True
                
        except Exception as e:
            logging.error(f"Error continuing assessment: {e}")
            self._handle_error(e, "continue_assessment")
            return False
    
    def set_callbacks(self,
                     on_question_start: Optional[Callable] = None,
                     on_question_complete: Optional[Callable] = None,
                     on_interruption: Optional[Callable] = None,
                     on_assessment_complete: Optional[Callable] = None) -> None:
        """
        Set callback functions for assessment events.
        
        Args:
            on_question_start: Called when a question starts presenting
            on_question_complete: Called when a question finishes presenting
            on_interruption: Called when an interruption is detected
            on_assessment_complete: Called when assessment is completed
        """
        self.on_question_start_callback = on_question_start
        self.on_question_complete_callback = on_question_complete
        self.on_interruption_callback = on_interruption
        self.on_assessment_complete_callback = on_assessment_complete
        logging.info("Assessment callbacks configured")
    
    def is_question_presentation_complete(self) -> bool:
        """
        Check if the current question presentation is complete.
        
        Returns:
            bool: True if question presentation is finished, False otherwise
        """
        return not self.is_presenting_question and self.current_question_completion_time is not None
    
    def get_question_presentation_duration(self, question_index: Optional[int] = None) -> Optional[float]:
        """
        Get the presentation duration for a specific question.
        
        Args:
            question_index: Index of question to check (defaults to current question)
            
        Returns:
            Optional[float]: Duration in seconds, or None if not available
        """
        if question_index is None:
            question_index = self.current_question_index
        
        if 0 <= question_index < len(self.responses):
            return self.responses[question_index].presentation_duration
        return None
    
    def get_time_since_question_completion(self) -> Optional[float]:
        """
        Get time elapsed since current question presentation completed.
        
        Returns:
            Optional[float]: Time in seconds, or None if question not completed
        """
        if self.current_question_completion_time:
            return time.time() - self.current_question_completion_time
        return None
    
    def force_proceed_to_next_question(self) -> bool:
        """
        Force proceeding to next question (useful for manual control).
        
        Returns:
            bool: True if successfully proceeded, False otherwise
        """
        try:
            # Cancel any active timeout
            self._cancel_response_timeout()
            
            # Mark current response as manually skipped if no response recorded
            if (self.responses and 
                self.current_question_index < len(self.responses) and 
                not self.responses[self.current_question_index].user_response):
                self.responses[self.current_question_index].user_response = "[Manually skipped]"
            
            return self.proceed_to_next_question()
            
        except Exception as e:
            logging.error(f"Error force proceeding to next question: {e}")
            return False

    # Session State Management Methods
    
    def save_session_state(self) -> bool:
        """
        Save current assessment session state to disk for recovery.
        
        Returns:
            bool: True if session saved successfully, False otherwise
        """
        try:
            if not self.session_id:
                self.session_id = f"session_{int(time.time())}_{os.getpid()}"
            
            self.session_file_path = os.path.join(self.session_storage_path, f"{self.session_id}.pkl")
            
            # Prepare session state data
            session_data = {
                "session_id": self.session_id,
                "state": self.state.value,
                "current_question_index": self.current_question_index,
                "interruption_count": self.interruption_count,
                "responses": self.responses,
                "assessment_start_time": self.assessment_start_time,
                "current_question_start_time": self.current_question_start_time,
                "current_question_completion_time": self.current_question_completion_time,
                "is_presenting_question": self.is_presenting_question,
                "current_assessment_id": self.current_assessment_id,
                "interruption_threshold": self.interruption_threshold,
                "question_timeout": self.question_timeout,
                "save_timestamp": time.time(),
                "error_count": self.error_count,
                "recovery_attempts": self.recovery_attempts,
                "error_log": self.error_log[-10:]  # Keep last 10 errors
            }
            
            # Save to file using pickle for complex objects
            with open(self.session_file_path, 'wb') as f:
                pickle.dump(session_data, f)
            
            self.last_save_time = time.time()
            logging.info(f"Session state saved to {self.session_file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error saving session state: {e}")
            self._handle_error(e, "save_session_state")
            return False
    
    def load_session_state(self, session_id: Optional[str] = None) -> bool:
        """
        Load assessment session state from disk for recovery.
        
        Args:
            session_id: Specific session ID to load (defaults to finding latest)
            
        Returns:
            bool: True if session loaded successfully, False otherwise
        """
        try:
            if session_id:
                session_file = os.path.join(self.session_storage_path, f"{session_id}.pkl")
            else:
                # Find the most recent session file
                session_file = self._find_latest_session_file()
            
            if not session_file or not os.path.exists(session_file):
                logging.warning(f"Session file not found: {session_file}")
                return False
            
            # Load session data
            with open(session_file, 'rb') as f:
                session_data = pickle.load(f)
            
            # Restore assessment state
            self.session_id = session_data.get("session_id")
            self.state = AssessmentState(session_data.get("state", "not_started"))
            self.current_question_index = session_data.get("current_question_index", 0)
            self.interruption_count = session_data.get("interruption_count", 0)
            self.responses = session_data.get("responses", [])
            self.assessment_start_time = session_data.get("assessment_start_time")
            self.current_question_start_time = session_data.get("current_question_start_time")
            self.current_question_completion_time = session_data.get("current_question_completion_time")
            self.is_presenting_question = session_data.get("is_presenting_question", False)
            self.current_assessment_id = session_data.get("current_assessment_id")
            self.error_count = session_data.get("error_count", 0)
            self.recovery_attempts = session_data.get("recovery_attempts", 0)
            self.error_log = session_data.get("error_log", [])
            
            self.session_file_path = session_file
            logging.info(f"Session state loaded from {session_file}")
            
            # If assessment was in progress, set up for recovery
            if self.state == AssessmentState.IN_PROGRESS:
                self.state = AssessmentState.RECOVERING
                logging.info("Assessment state set to RECOVERING - ready for continuation")
            
            return True
            
        except Exception as e:
            logging.error(f"Error loading session state: {e}")
            self._handle_error(e, "load_session_state")
            return False
    
    def _find_latest_session_file(self) -> Optional[str]:
        """
        Find the most recent session file in the session storage directory.
        
        Returns:
            Optional[str]: Path to the latest session file or None if none found
        """
        try:
            if not os.path.exists(self.session_storage_path):
                return None
            
            session_files = [
                f for f in os.listdir(self.session_storage_path) 
                if f.startswith("session_") and f.endswith(".pkl")
            ]
            
            if not session_files:
                return None
            
            # Sort by modification time (newest first)
            session_files.sort(
                key=lambda f: os.path.getmtime(os.path.join(self.session_storage_path, f)),
                reverse=True
            )
            
            return os.path.join(self.session_storage_path, session_files[0])
            
        except Exception as e:
            logging.error(f"Error finding latest session file: {e}")
            return None
    
    def delete_session_state(self, session_id: Optional[str] = None) -> bool:
        """
        Delete session state file.
        
        Args:
            session_id: Session ID to delete (defaults to current session)
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            if session_id:
                session_file = os.path.join(self.session_storage_path, f"{session_id}.pkl")
            else:
                session_file = self.session_file_path
            
            if session_file and os.path.exists(session_file):
                os.remove(session_file)
                logging.info(f"Session state deleted: {session_file}")
                
                if session_file == self.session_file_path:
                    self.session_file_path = None
                    self.session_id = None
                
                return True
            else:
                logging.warning(f"Session file not found for deletion: {session_file}")
                return False
                
        except Exception as e:
            logging.error(f"Error deleting session state: {e}")
            return False
    
    def list_available_sessions(self) -> List[Dict]:
        """
        List all available session files with metadata.
        
        Returns:
            List[Dict]: List of session information dictionaries
        """
        try:
            sessions = []
            
            if not os.path.exists(self.session_storage_path):
                return sessions
            
            for filename in os.listdir(self.session_storage_path):
                if filename.startswith("session_") and filename.endswith(".pkl"):
                    filepath = os.path.join(self.session_storage_path, filename)
                    try:
                        # Get file metadata
                        stat = os.stat(filepath)
                        
                        # Try to load basic session info
                        with open(filepath, 'rb') as f:
                            session_data = pickle.load(f)
                        
                        session_info = {
                            "session_id": session_data.get("session_id", filename[:-4]),
                            "filename": filename,
                            "filepath": filepath,
                            "state": session_data.get("state", "unknown"),
                            "current_question": session_data.get("current_question_index", 0) + 1,
                            "interruption_count": session_data.get("interruption_count", 0),
                            "save_timestamp": session_data.get("save_timestamp", stat.st_mtime),
                            "file_size": stat.st_size,
                            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        }
                        sessions.append(session_info)
                        
                    except Exception as e:
                        logging.warning(f"Error reading session file {filename}: {e}")
                        continue
            
            # Sort by save timestamp (newest first)
            sessions.sort(key=lambda x: x.get("save_timestamp", 0), reverse=True)
            return sessions
            
        except Exception as e:
            logging.error(f"Error listing available sessions: {e}")
            return []
    
    def start_auto_save(self) -> None:
        """Start automatic session state saving."""
        try:
            self._stop_auto_save()  # Stop any existing timer
            
            if self.auto_save_interval > 0:
                self.auto_save_timer = threading.Timer(
                    self.auto_save_interval,
                    self._auto_save_callback
                )
                self.auto_save_timer.start()
                logging.info(f"Auto-save started with {self.auto_save_interval}s interval")
            
        except Exception as e:
            logging.error(f"Error starting auto-save: {e}")
    
    def _stop_auto_save(self) -> None:
        """Stop automatic session state saving."""
        if self.auto_save_timer:
            self.auto_save_timer.cancel()
            self.auto_save_timer = None
            logging.debug("Auto-save stopped")
    
    def _auto_save_callback(self) -> None:
        """Callback for automatic session saving."""
        try:
            if self.state in [AssessmentState.IN_PROGRESS, AssessmentState.PAUSED]:
                self.save_session_state()
            
            # Schedule next auto-save
            if self.auto_save_interval > 0:
                self.auto_save_timer = threading.Timer(
                    self.auto_save_interval,
                    self._auto_save_callback
                )
                self.auto_save_timer.start()
                
        except Exception as e:
            logging.error(f"Error in auto-save callback: {e}")
            self._handle_error(e, "auto_save_callback")


# Example usage and testing
if __name__ == "__main__":
    # Create assessment instance with custom timeout
    assessment = InteractiveADHDAssessment(question_timeout=5)  # 5 second timeout for testing
    
    # Example callback functions
    def on_question_start(question_index, question_text):
        print(f"Question {question_index + 1} started: {question_text[:50]}...")
        print(f"Presentation timing tracking active...")
    
    def on_question_complete(question_index):
        duration = assessment.get_question_presentation_duration(question_index)
        print(f"Question {question_index + 1} presentation completed in {duration:.2f}s")
        print(f"Waiting for user response (timeout: {assessment.question_timeout}s)...")
    
    def on_interruption(count, timestamp):
        print(f"Interruption detected! Total: {count}")
        time_since_start = timestamp - assessment.current_question_start_time if assessment.current_question_start_time else 0
        print(f"Interruption occurred {time_since_start:.2f}s after question started")
    
    def on_complete(results):
        print(f"\nAssessment complete!")
        print(f"ADHD flagged: {results['adhd_flagged']}")
        print(f"Total interruptions: {results['interruption_count']}")
        print(f"Timing statistics: {results['timing_statistics']}")
    
    # Set callbacks
    assessment.set_callbacks(
        on_question_start=on_question_start,
        on_question_complete=on_question_complete,
        on_interruption=on_interruption,
        on_assessment_complete=on_complete
    )
    
    # Example assessment flow (would be integrated with actual speech recognition)
    print("Starting assessment with timing system...")
    assessment.start_assessment()
    
    # Simulate some responses and interruptions with timing
    print("\nSimulating user interactions...")
    
    # Wait a bit then simulate an interruption
    time.sleep(2)
    print("Simulating interruption during question presentation...")
    assessment.handle_user_response("Wait, let me interrupt this question!")
    
    # Check timing functionality
    print(f"\nQuestion presentation complete: {assessment.is_question_presentation_complete()}")
    print(f"Time since completion: {assessment.get_time_since_question_completion()}")
    
    # Get current state
    state = assessment.get_current_state()
    print(f"Current state: {state}")
    
    # Demonstrate timeout handling by not responding to next question
    print("\nLetting next question timeout...")
    time.sleep(6)  # Wait longer than timeout
    
    print("Assessment timing system demonstration complete.")  
  # Error Handling and Recovery Methods
    
    def _handle_error(self, error: Exception, operation: str, retry: bool = True) -> bool:
        """
        Handle errors with logging, recovery attempts, and user notification.
        
        Args:
            error: The exception that occurred
            operation: Name of the operation that failed
            retry: Whether to attempt retry for recoverable errors
            
        Returns:
            bool: True if error was handled successfully, False otherwise
        """
        try:
            self.error_count += 1
            self.last_error_time = time.time()
            
            # Log error details
            error_info = {
                "timestamp": time.time(),
                "operation": operation,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "assessment_state": self.state.value,
                "current_question": self.current_question_index,
                "error_count": self.error_count
            }
            
            self.error_log.append(error_info)
            logging.error(f"Error in {operation}: {error} (Error #{self.error_count})")
            
            # Determine if error is recoverable
            is_recoverable = self._is_recoverable_error(error)
            
            # Handle specific error types
            if isinstance(error, AudioSystemError):
                return self._handle_audio_error(error, retry)
            elif isinstance(error, UserDisconnectionError):
                return self._handle_disconnection_error(error)
            else:
                return self._handle_general_error(error, operation, is_recoverable, retry)
                
        except Exception as e:
            logging.critical(f"Error in error handler: {e}")
            return False
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """
        Determine if an error is recoverable.
        
        Args:
            error: The exception to check
            
        Returns:
            bool: True if error is recoverable, False otherwise
        """
        recoverable_errors = [
            ConnectionError,
            TimeoutError,
            OSError,
            AudioSystemError
        ]
        
        # Check if error type is recoverable
        for recoverable_type in recoverable_errors:
            if isinstance(error, recoverable_type):
                return True
        
        # Check error message for recoverable patterns
        error_msg = str(error).lower()
        recoverable_patterns = [
            "connection",
            "timeout",
            "network",
            "audio",
            "microphone",
            "speaker",
            "device"
        ]
        
        return any(pattern in error_msg for pattern in recoverable_patterns)
    
    def _handle_audio_error(self, error: AudioSystemError, retry: bool = True) -> bool:
        """
        Handle audio system errors with specific recovery strategies.
        
        Args:
            error: The audio system error
            retry: Whether to attempt retry
            
        Returns:
            bool: True if handled successfully, False otherwise
        """
        try:
            logging.warning(f"Audio system error detected: {error}")
            
            if not retry or self.recovery_attempts >= self.max_retry_attempts:
                logging.error("Max audio recovery attempts reached")
                self._set_error_state("Audio system failure - max retries exceeded")
                return False
            
            self.recovery_attempts += 1
            self.is_recovering = True
            
            # Save current state before attempting recovery
            self.save_session_state()
            
            # Attempt audio system recovery
            recovery_success = self._attempt_audio_recovery()
            
            if recovery_success:
                logging.info("Audio system recovery successful")
                self.recovery_attempts = 0
                self.is_recovering = False
                
                # Trigger recovery callback
                if self.on_recovery_callback:
                    self.on_recovery_callback("audio_recovery_success", self.get_current_state())
                
                return True
            else:
                logging.warning(f"Audio recovery attempt {self.recovery_attempts} failed")
                
                # Try alternative audio approach
                if self.recovery_attempts < self.max_retry_attempts:
                    return self._try_alternative_audio_approach()
                else:
                    self._set_error_state("Audio system recovery failed")
                    return False
                    
        except Exception as e:
            logging.error(f"Error in audio error handler: {e}")
            return False
    
    def _attempt_audio_recovery(self) -> bool:
        """
        Attempt to recover from audio system failures.
        
        Returns:
            bool: True if recovery successful, False otherwise
        """
        try:
            # Wait a moment for system to stabilize
            time.sleep(2)
            
            # Test audio output with a simple message
            test_message = "Audio system recovery test"
            text_to_speech_with_elevenlabs(test_message, self.audio_output_path)
            
            # If we get here without exception, audio is working
            logging.info("Audio system test successful")
            return True
            
        except Exception as e:
            logging.warning(f"Audio recovery test failed: {e}")
            return False
    
    def _try_alternative_audio_approach(self) -> bool:
        """
        Try alternative audio approaches when primary system fails.
        
        Returns:
            bool: True if alternative approach works, False otherwise
        """
        try:
            # Try using a different audio file format
            alternative_path = self.audio_output_path.replace('.wav', '_alt.mp3')
            
            # Attempt with alternative path
            test_message = "Alternative audio system test"
            text_to_speech_with_elevenlabs(test_message, alternative_path)
            
            # Update audio path if successful
            self.audio_output_path = alternative_path
            logging.info("Alternative audio approach successful")
            return True
            
        except Exception as e:
            logging.warning(f"Alternative audio approach failed: {e}")
            return False
    
    def _handle_disconnection_error(self, error: UserDisconnectionError) -> bool:
        """
        Handle user disconnection errors with session preservation.
        
        Args:
            error: The disconnection error
            
        Returns:
            bool: True if handled successfully, False otherwise
        """
        try:
            logging.warning(f"User disconnection detected: {error}")
            
            # Immediately save session state
            save_success = self.save_session_state()
            
            if save_success:
                logging.info("Session state preserved for reconnection")
                
                # Pause assessment to allow for reconnection
                if self.state == AssessmentState.IN_PROGRESS:
                    self.pause_assessment()
                
                # Trigger disconnection callback
                if self.on_error_callback:
                    self.on_error_callback("user_disconnection", {
                        "session_id": self.session_id,
                        "current_question": self.current_question_index + 1,
                        "progress": self.get_current_state()
                    })
                
                return True
            else:
                logging.error("Failed to save session state during disconnection")
                return False
                
        except Exception as e:
            logging.error(f"Error handling disconnection: {e}")
            return False
    
    def _handle_general_error(self, error: Exception, operation: str, is_recoverable: bool, retry: bool) -> bool:
        """
        Handle general errors with appropriate recovery strategies.
        
        Args:
            error: The exception that occurred
            operation: Name of the failed operation
            is_recoverable: Whether the error is recoverable
            retry: Whether to attempt retry
            
        Returns:
            bool: True if handled successfully, False otherwise
        """
        try:
            if not is_recoverable:
                logging.error(f"Non-recoverable error in {operation}: {error}")
                self._set_error_state(f"Non-recoverable error: {error}")
                return False
            
            if not retry or self.recovery_attempts >= self.max_retry_attempts:
                logging.error(f"Max recovery attempts reached for {operation}")
                self._set_error_state(f"Recovery failed for {operation}")
                return False
            
            self.recovery_attempts += 1
            self.is_recovering = True
            
            # Save state before recovery attempt
            self.save_session_state()
            
            # Wait before retry
            retry_delay = min(2 ** self.recovery_attempts, 10)  # Exponential backoff, max 10s
            logging.info(f"Retrying {operation} in {retry_delay} seconds (attempt {self.recovery_attempts})")
            time.sleep(retry_delay)
            
            # Trigger recovery callback
            if self.on_recovery_callback:
                self.on_recovery_callback("general_recovery_attempt", {
                    "operation": operation,
                    "attempt": self.recovery_attempts,
                    "error": str(error)
                })
            
            return True
            
        except Exception as e:
            logging.error(f"Error in general error handler: {e}")
            return False
    
    def _set_error_state(self, error_message: str) -> None:
        """
        Set assessment to error state with appropriate cleanup.
        
        Args:
            error_message: Description of the error
        """
        try:
            self.state = AssessmentState.ERROR
            self.is_recovering = False
            
            # Cancel any active timers
            self._cancel_response_timeout()
            self._stop_auto_save()
            
            # Save final state
            self.save_session_state()
            
            logging.error(f"Assessment set to ERROR state: {error_message}")
            
            # Trigger error callback
            if self.on_error_callback:
                self.on_error_callback("assessment_error", {
                    "error_message": error_message,
                    "error_count": self.error_count,
                    "session_id": self.session_id,
                    "current_state": self.get_current_state()
                })
                
        except Exception as e:
            logging.critical(f"Error setting error state: {e}")
    
    def recover_from_error(self) -> bool:
        """
        Attempt to recover from error state and continue assessment.
        
        Returns:
            bool: True if recovery successful, False otherwise
        """
        try:
            if self.state != AssessmentState.ERROR:
                logging.warning("Recovery attempted but assessment not in error state")
                return False
            
            logging.info("Attempting recovery from error state")
            self.state = AssessmentState.RECOVERING
            self.is_recovering = True
            
            # Reset recovery attempts for fresh start
            self.recovery_attempts = 0
            
            # Test basic functionality
            recovery_tests = [
                self._test_audio_system,
                self._test_file_system,
                self._test_session_state
            ]
            
            for test in recovery_tests:
                if not test():
                    logging.error(f"Recovery test failed: {test.__name__}")
                    self._set_error_state("Recovery test failed")
                    return False
            
            # If all tests pass, restore to previous state
            if self.current_question_index < len(self.ASSESSMENT_QUESTIONS):
                self.state = AssessmentState.IN_PROGRESS
                logging.info("Recovery successful - assessment resumed")
                
                # Restart auto-save
                self.start_auto_save()
                
                # Trigger recovery callback
                if self.on_recovery_callback:
                    self.on_recovery_callback("full_recovery_success", self.get_current_state())
                
                return True
            else:
                # Assessment was complete, just mark as completed
                self.state = AssessmentState.COMPLETED
                logging.info("Recovery successful - assessment was already complete")
                return True
                
        except Exception as e:
            logging.error(f"Error during recovery: {e}")
            self._handle_error(e, "recover_from_error", retry=False)
            return False
    
    def _test_audio_system(self) -> bool:
        """Test audio system functionality."""
        try:
            test_message = "Recovery audio test"
            text_to_speech_with_elevenlabs(test_message, self.audio_output_path)
            return True
        except Exception as e:
            logging.warning(f"Audio system test failed: {e}")
            return False
    
    def _test_file_system(self) -> bool:
        """Test file system access."""
        try:
            # Test write access to results directory
            test_file = os.path.join(self.results_storage_path, "recovery_test.tmp")
            with open(test_file, 'w') as f:
                f.write("recovery test")
            os.remove(test_file)
            return True
        except Exception as e:
            logging.warning(f"File system test failed: {e}")
            return False
    
    def _test_session_state(self) -> bool:
        """Test session state functionality."""
        try:
            # Test session save/load
            original_error_count = self.error_count
            self.error_count += 1  # Temporary change
            
            if self.save_session_state():
                self.error_count = original_error_count  # Restore
                return True
            return False
        except Exception as e:
            logging.warning(f"Session state test failed: {e}")
            return False
    
    def get_error_summary(self) -> Dict:
        """
        Get summary of errors and recovery attempts.
        
        Returns:
            Dict: Error summary information
        """
        return {
            "total_errors": self.error_count,
            "recovery_attempts": self.recovery_attempts,
            "is_recovering": self.is_recovering,
            "last_error_time": self.last_error_time,
            "recent_errors": self.error_log[-5:] if self.error_log else [],
            "error_rate": self.error_count / max(1, time.time() - (self.assessment_start_time or time.time())),
            "recovery_success_rate": max(0, 1 - (self.recovery_attempts / max(1, self.error_count)))
        }
    
    def cleanup_old_sessions(self, max_age_days: int = 7) -> int:
        """
        Clean up old session files to prevent storage bloat.
        
        Args:
            max_age_days: Maximum age of session files to keep
            
        Returns:
            int: Number of files cleaned up
        """
        try:
            if not os.path.exists(self.session_storage_path):
                return 0
            
            cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
            cleaned_count = 0
            
            for filename in os.listdir(self.session_storage_path):
                if filename.startswith("session_") and filename.endswith(".pkl"):
                    filepath = os.path.join(self.session_storage_path, filename)
                    
                    try:
                        if os.path.getmtime(filepath) < cutoff_time:
                            os.remove(filepath)
                            cleaned_count += 1
                            logging.debug(f"Cleaned up old session file: {filename}")
                    except Exception as e:
                        logging.warning(f"Error cleaning up session file {filename}: {e}")
            
            if cleaned_count > 0:
                logging.info(f"Cleaned up {cleaned_count} old session files")
            
            return cleaned_count
            
        except Exception as e:
            logging.error(f"Error during session cleanup: {e}")
            return 0