"""
Unit tests for InteractiveADHDAssessment class

This module contains comprehensive unit tests for the InteractiveADHDAssessment class,
focusing on core functionality, timing accuracy, interruption detection, and result calculation.
"""

import unittest
import time
import os
import tempfile
import shutil
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from interactive_assessment import (
    InteractiveADHDAssessment,
    AssessmentState,
    QuestionResponse,
    AssessmentError,
    AudioSystemError,
    UserDisconnectionError
)


class TestInteractiveADHDAssessment(unittest.TestCase):
    """Test cases for InteractiveADHDAssessment class methods"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.results_path = os.path.join(self.temp_dir, "results")
        self.sessions_path = os.path.join(self.temp_dir, "sessions")
        self.audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        
        # Create assessment instance with test configuration
        self.assessment = InteractiveADHDAssessment(
            interruption_threshold=3,
            question_timeout=2,  # Short timeout for faster tests
            audio_output_path=self.audio_path,
            results_storage_path=self.results_path,
            session_storage_path=self.sessions_path,
            auto_save_interval=1,  # Short interval for testing
            max_retry_attempts=2
        )
        
        # Mock callbacks for testing
        self.mock_callbacks = {
            'on_question_start': Mock(),
            'on_question_complete': Mock(),
            'on_interruption': Mock(),
            'on_assessment_complete': Mock()
        }
        
        self.assessment.set_callbacks(**self.mock_callbacks)
    
    def tearDown(self):
        """Clean up after each test method"""
        # Reset assessment state
        self.assessment.reset_assessment()
        
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_assessment_initialization(self):
        """Test proper initialization of assessment instance"""
        self.assertEqual(self.assessment.state, AssessmentState.NOT_STARTED)
        self.assertEqual(self.assessment.current_question_index, 0)
        self.assertEqual(self.assessment.interruption_count, 0)
        self.assertEqual(self.assessment.interruption_threshold, 3)
        self.assertEqual(self.assessment.question_timeout, 2)
        self.assertEqual(len(self.assessment.responses), 0)
        self.assertIsNone(self.assessment.assessment_start_time)
        self.assertFalse(self.assessment.is_presenting_question)
    
    @patch('interactive_assessment.text_to_speech_with_elevenlabs')
    def test_start_assessment_success(self, mock_tts):
        """Test successful assessment start"""
        mock_tts.return_value = None
        
        result = self.assessment.start_assessment()
        
        self.assertTrue(result)
        self.assertEqual(self.assessment.state, AssessmentState.IN_PROGRESS)
        self.assertIsNotNone(self.assessment.assessment_start_time)
        self.assertEqual(self.assessment.current_question_index, 0)
        self.assertIsNotNone(self.assessment.session_id)
        
        # Verify first question is being presented
        self.assertTrue(len(self.assessment.responses) > 0)
        self.assertEqual(self.assessment.responses[0].question_index, 0)
    
    def test_start_assessment_already_in_progress(self):
        """Test starting assessment when already in progress"""
        self.assessment.state = AssessmentState.IN_PROGRESS
        
        result = self.assessment.start_assessment()
        
        self.assertFalse(result)
    
    @patch('interactive_assessment.text_to_speech_with_elevenlabs')
    def test_present_current_question(self, mock_tts):
        """Test question presentation functionality"""
        mock_tts.return_value = None
        
        # Start assessment first
        self.assessment.start_assessment()
        
        # Wait for presentation thread to complete
        if self.assessment.presentation_thread:
            self.assessment.presentation_thread.join(timeout=3)
        
        # Verify question was presented
        self.assertEqual(len(self.assessment.responses), 1)
        response = self.assessment.responses[0]
        self.assertEqual(response.question_index, 0)
        self.assertEqual(response.question_text, self.assessment.ASSESSMENT_QUESTIONS[0])
        self.assertIsNotNone(response.start_time)
        
        # Verify callback was called
        self.mock_callbacks['on_question_start'].assert_called_once()
    
    def test_present_question_beyond_range(self):
        """Test presenting question when index is beyond available questions"""
        self.assessment.current_question_index = len(self.assessment.ASSESSMENT_QUESTIONS)
        
        result = self.assessment.present_current_question()
        
        self.assertFalse(result)
    
    @patch('interactive_assessment.text_to_speech_with_elevenlabs')
    def test_handle_user_response_normal(self, mock_tts):
        """Test handling normal user response after question completion"""
        mock_tts.return_value = None
        
        # Start assessment and wait for question to complete
        self.assessment.start_assessment()
        if self.assessment.presentation_thread:
            self.assessment.presentation_thread.join(timeout=3)
        
        # Simulate user response after question completion
        response_text = "This is my response to the question"
        result = self.assessment.handle_user_response(response_text)
        
        self.assertTrue(result)
        self.assertEqual(self.assessment.responses[0].user_response, response_text)
        self.assertFalse(self.assessment.responses[0].was_interrupted)
        self.assertEqual(self.assessment.current_question_index, 1)
    
    @patch('interactive_assessment.text_to_speech_with_elevenlabs')
    def test_handle_user_response_interruption(self, mock_tts):
        """Test handling user response during question presentation (interruption)"""
        mock_tts.return_value = None
        
        # Start assessment
        self.assessment.start_assessment()
        
        # Simulate interruption during question presentation
        self.assessment.is_presenting_question = True
        response_text = "I'm interrupting this question!"
        interruption_time = time.time()
        
        result = self.assessment.handle_user_response(response_text, interruption_time)
        
        self.assertTrue(result)
        self.assertEqual(self.assessment.interruption_count, 1)
        self.assertTrue(self.assessment.responses[0].was_interrupted)
        self.assertEqual(self.assessment.responses[0].user_response, response_text)
        self.assertIsNotNone(self.assessment.responses[0].interruption_time)
        
        # Verify interruption callback was called
        self.mock_callbacks['on_interruption'].assert_called_once()
    
    def test_record_interruption(self):
        """Test interruption recording functionality"""
        # Add a response to work with
        response = QuestionResponse(
            question_index=0,
            question_text="Test question",
            start_time=time.time(),
            completion_time=None,
            user_response=None,
            was_interrupted=False,
            interruption_time=None
        )
        self.assessment.responses.append(response)
        
        interruption_time = time.time()
        self.assessment.record_interruption(interruption_time)
        
        self.assertEqual(self.assessment.interruption_count, 1)
        self.assertTrue(self.assessment.responses[0].was_interrupted)
        self.assertEqual(self.assessment.responses[0].interruption_time, interruption_time)
        
        # Verify callback was called
        self.mock_callbacks['on_interruption'].assert_called_once_with(1, interruption_time)
    
    @patch('interactive_assessment.text_to_speech_with_elevenlabs')
    def test_proceed_to_next_question(self, mock_tts):
        """Test proceeding to next question"""
        mock_tts.return_value = None
        
        # Start assessment
        self.assessment.start_assessment()
        if self.assessment.presentation_thread:
            self.assessment.presentation_thread.join(timeout=3)
        
        # Proceed to next question
        result = self.assessment.proceed_to_next_question()
        
        self.assertTrue(result)
        self.assertEqual(self.assessment.current_question_index, 1)
    
    @patch('interactive_assessment.text_to_speech_with_elevenlabs')
    def test_complete_assessment_when_all_questions_done(self, mock_tts):
        """Test assessment completion when all questions are finished"""
        mock_tts.return_value = None
        
        # Set up assessment at the last question
        self.assessment.current_question_index = len(self.assessment.ASSESSMENT_QUESTIONS) - 1
        self.assessment.state = AssessmentState.IN_PROGRESS
        
        result = self.assessment.proceed_to_next_question()
        
        self.assertTrue(result)
        self.assertEqual(self.assessment.state, AssessmentState.COMPLETED)
        
        # Verify completion callback was called
        self.mock_callbacks['on_assessment_complete'].assert_called_once()


class TestQuestionTimingAndInterruption(unittest.TestCase):
    """Test cases for question timing and interruption detection accuracy"""
    
    def setUp(self):
        """Set up test fixtures for timing tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.assessment = InteractiveADHDAssessment(
            interruption_threshold=3,
            question_timeout=1,  # Very short for testing
            results_storage_path=os.path.join(self.temp_dir, "results"),
            session_storage_path=os.path.join(self.temp_dir, "sessions")
        )
    
    def tearDown(self):
        """Clean up after timing tests"""
        self.assessment.reset_assessment()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('interactive_assessment.text_to_speech_with_elevenlabs')
    def test_question_timing_accuracy(self, mock_tts):
        """Test accuracy of question presentation timing"""
        # Mock TTS to take a known amount of time
        def mock_tts_delay(*args, **kwargs):
            time.sleep(0.5)  # Simulate 0.5 second TTS duration
        
        mock_tts.side_effect = mock_tts_delay
        
        start_time = time.time()
        self.assessment.start_assessment()
        
        # Wait for presentation to complete
        if self.assessment.presentation_thread:
            self.assessment.presentation_thread.join(timeout=3)
        
        # Check timing accuracy
        response = self.assessment.responses[0]
        self.assertIsNotNone(response.start_time)
        self.assertIsNotNone(response.completion_time)
        self.assertIsNotNone(response.presentation_duration)
        
        # Verify timing is reasonable (0.4-0.6 seconds allowing for some variance)
        self.assertGreater(response.presentation_duration, 0.4)
        self.assertLess(response.presentation_duration, 0.6)
        
        # Verify start time is after our test start
        self.assertGreaterEqual(response.start_time, start_time)
        
        # Verify completion time is after start time
        self.assertGreater(response.completion_time, response.start_time)
    
    @patch('interactive_assessment.text_to_speech_with_elevenlabs')
    def test_interruption_timing_detection(self, mock_tts):
        """Test accurate detection of interruption timing"""
        mock_tts.return_value = None
        
        # Start assessment
        self.assessment.start_assessment()
        
        # Simulate interruption at specific time
        question_start = self.assessment.current_question_start_time
        interruption_delay = 0.3  # 300ms after question start
        interruption_time = question_start + interruption_delay
        
        # Force interruption state and record
        self.assessment.is_presenting_question = True
        self.assessment.record_interruption(interruption_time)
        
        # Verify interruption timing
        response = self.assessment.responses[0]
        calculated_delay = response.interruption_time - response.start_time
        
        self.assertAlmostEqual(calculated_delay, interruption_delay, places=2)
        self.assertTrue(response.was_interrupted)
    
    @patch('interactive_assessment.text_to_speech_with_elevenlabs')
    def test_multiple_interruption_detection(self, mock_tts):
        """Test detection of multiple interruptions across questions"""
        mock_tts.return_value = None
        
        # Start assessment
        self.assessment.start_assessment()
        
        # Simulate interruptions on first 3 questions
        for i in range(3):
            if i < len(self.assessment.responses):
                self.assessment.is_presenting_question = True
                self.assessment.record_interruption(time.time())
                
                # Move to next question
                if i < 2:  # Don't proceed after last interruption
                    self.assessment.proceed_to_next_question()
        
        # Verify all interruptions were recorded
        self.assertEqual(self.assessment.interruption_count, 3)
        
        # Verify each response shows interruption
        for i in range(min(3, len(self.assessment.responses))):
            self.assertTrue(self.assessment.responses[i].was_interrupted)
            self.assertIsNotNone(self.assessment.responses[i].interruption_time)
    
    @patch('interactive_assessment.text_to_speech_with_elevenlabs')
    def test_response_timeout_handling(self, mock_tts):
        """Test handling of response timeouts"""
        mock_tts.return_value = None
        
        # Start assessment with very short timeout
        self.assessment.question_timeout = 0.5
        self.assessment.start_assessment()
        
        # Wait for presentation to complete
        if self.assessment.presentation_thread:
            self.assessment.presentation_thread.join(timeout=3)
        
        # Wait for timeout to occur
        time.sleep(0.7)
        
        # Verify timeout was handled
        if len(self.assessment.responses) > 0:
            response = self.assessment.responses[0]
            # Timeout handling may set specific response text
            self.assertTrue(
                response.timeout_occurred or 
                response.user_response == "[No response - timeout]" or
                self.assessment.current_question_index > 0  # Moved to next question
            )


class TestResultCalculationLogic(unittest.TestCase):
    """Test cases for result calculation logic and edge cases"""
    
    def setUp(self):
        """Set up test fixtures for result calculation tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.assessment = InteractiveADHDAssessment(
            interruption_threshold=3,
            results_storage_path=os.path.join(self.temp_dir, "results"),
            session_storage_path=os.path.join(self.temp_dir, "sessions")
        )
    
    def tearDown(self):
        """Clean up after result calculation tests"""
        self.assessment.reset_assessment()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_calculate_results_no_adhd_flagged(self):
        """Test result calculation when ADHD should not be flagged"""
        # Set up assessment with 2 interruptions (below threshold of 3)
        self.assessment.interruption_count = 2
        self.assessment.interruption_threshold = 3
        self.assessment.assessment_start_time = time.time() - 100
        
        # Add some responses
        for i in range(5):
            response = QuestionResponse(
                question_index=i,
                question_text=f"Question {i+1}",
                start_time=time.time() - 90 + (i * 10),
                completion_time=time.time() - 85 + (i * 10),
                user_response=f"Response {i+1}",
                was_interrupted=(i < 2),  # First 2 are interrupted
                interruption_time=time.time() - 88 + (i * 10) if i < 2 else None
            )
            self.assessment.responses.append(response)
        
        results = self.assessment.calculate_results()
        
        # Verify core results - Requirements 3.1, 3.2
        self.assertFalse(results['adhd_flagged'])
        self.assertEqual(results['interruption_count'], 2)
        self.assertEqual(results['interruption_threshold'], 3)
        self.assertFalse(results['flag_threshold_met'])
        
        # Verify completion metrics
        self.assertEqual(results['total_questions'], 5)
        self.assertEqual(results['completed_questions'], 5)
        self.assertEqual(results['completion_rate'], 1.0)
        self.assertTrue(results['assessment_completed'])
        
        # Verify message content
        self.assertIn("No ADHD indicators detected", results['assessment_message'])
        self.assertIn("2 interruptions recorded", results['assessment_message'])
    
    def test_calculate_results_adhd_flagged(self):
        """Test result calculation when ADHD should be flagged"""
        # Set up assessment with 4 interruptions (above threshold of 3)
        self.assessment.interruption_count = 4
        self.assessment.interruption_threshold = 3
        self.assessment.assessment_start_time = time.time() - 100
        
        # Add responses with interruptions
        for i in range(5):
            response = QuestionResponse(
                question_index=i,
                question_text=f"Question {i+1}",
                start_time=time.time() - 90 + (i * 10),
                completion_time=time.time() - 85 + (i * 10),
                user_response=f"Response {i+1}",
                was_interrupted=(i < 4),  # First 4 are interrupted
                interruption_time=time.time() - 88 + (i * 10) if i < 4 else None
            )
            self.assessment.responses.append(response)
        
        results = self.assessment.calculate_results()
        
        # Verify core results - Requirements 3.1, 3.2
        self.assertTrue(results['adhd_flagged'])
        self.assertEqual(results['interruption_count'], 4)
        self.assertEqual(results['interruption_threshold'], 3)
        self.assertTrue(results['flag_threshold_met'])
        
        # Verify message content
        self.assertIn("ADHD indicators detected", results['assessment_message'])
        self.assertIn("4 interruptions recorded", results['assessment_message'])
        self.assertIn("Consider consulting a healthcare professional", results['assessment_message'])
    
    def test_calculate_results_edge_case_exact_threshold(self):
        """Test result calculation at exact threshold boundary"""
        # Set up assessment with exactly 3 interruptions (at threshold)
        self.assessment.interruption_count = 3
        self.assessment.interruption_threshold = 3
        
        results = self.assessment.calculate_results()
        
        # At threshold should flag ADHD - Requirements 1.5
        self.assertTrue(results['adhd_flagged'])
        self.assertTrue(results['flag_threshold_met'])
        self.assertEqual(results['interruption_count'], 3)
    
    def test_calculate_results_partial_completion(self):
        """Test result calculation with partial assessment completion"""
        # Set up assessment with only 3 out of 5 questions completed
        self.assessment.interruption_count = 2
        self.assessment.assessment_start_time = time.time() - 60
        
        # Add only 3 responses
        for i in range(3):
            response = QuestionResponse(
                question_index=i,
                question_text=f"Question {i+1}",
                start_time=time.time() - 50 + (i * 10),
                completion_time=time.time() - 45 + (i * 10),
                user_response=f"Response {i+1}",
                was_interrupted=(i < 2),
                interruption_time=time.time() - 48 + (i * 10) if i < 2 else None
            )
            self.assessment.responses.append(response)
        
        results = self.assessment.calculate_results()
        
        # Verify partial completion metrics
        self.assertEqual(results['total_questions'], 5)
        self.assertEqual(results['completed_questions'], 3)
        self.assertEqual(results['completion_rate'], 0.6)
        self.assertFalse(results['assessment_completed'])
    
    def test_calculate_results_no_responses(self):
        """Test result calculation with no responses (edge case)"""
        # Empty assessment
        self.assessment.interruption_count = 0
        self.assessment.responses = []
        
        results = self.assessment.calculate_results()
        
        # Verify safe handling of empty state
        self.assertFalse(results['adhd_flagged'])
        self.assertEqual(results['interruption_count'], 0)
        self.assertEqual(results['completed_questions'], 0)
        self.assertEqual(results['completion_rate'], 0)
        self.assertFalse(results['assessment_completed'])
    
    def test_calculate_results_timing_statistics(self):
        """Test calculation of detailed timing statistics"""
        self.assessment.assessment_start_time = time.time() - 120
        
        # Add responses with varied timing patterns
        base_time = time.time() - 100
        for i in range(3):
            response = QuestionResponse(
                question_index=i,
                question_text=f"Question {i+1}",
                start_time=base_time + (i * 20),
                completion_time=base_time + (i * 20) + 5,
                user_response=f"Response {i+1}",
                was_interrupted=(i == 1),  # Only middle question interrupted
                interruption_time=base_time + (i * 20) + 2 if i == 1 else None,
                presentation_duration=5.0
            )
            self.assessment.responses.append(response)
        
        self.assessment.interruption_count = 1
        
        results = self.assessment.calculate_results()
        
        # Verify timing statistics are calculated
        timing_stats = results.get('timing_statistics', {})
        self.assertIsInstance(timing_stats, dict)
        self.assertEqual(timing_stats.get('total_questions_presented'), 3)
        self.assertEqual(timing_stats.get('questions_with_interruptions'), 1)
        self.assertEqual(timing_stats.get('average_presentation_duration'), 5.0)
        self.assertGreater(timing_stats.get('total_presentation_time'), 0)
    
    def test_calculate_results_behavioral_analysis(self):
        """Test calculation of behavioral analysis patterns"""
        # Set up responses with specific interruption patterns
        base_time = time.time() - 100
        interruption_delays = [1.0, 3.0, 6.0]  # Early, mid, late interruptions
        
        for i in range(3):
            response = QuestionResponse(
                question_index=i,
                question_text=f"Question {i+1}",
                start_time=base_time + (i * 20),
                completion_time=base_time + (i * 20) + 10,
                user_response=f"Response {i+1}",
                was_interrupted=True,
                interruption_time=base_time + (i * 20) + interruption_delays[i]
            )
            self.assessment.responses.append(response)
        
        self.assessment.interruption_count = 3
        
        results = self.assessment.calculate_results()
        
        # Verify behavioral analysis
        behavioral = results.get('behavioral_analysis', {})
        self.assertIsInstance(behavioral, dict)
        
        timing_analysis = behavioral.get('interruption_timing', {})
        self.assertEqual(timing_analysis.get('early_interruptions'), 1)  # ≤2s
        self.assertEqual(timing_analysis.get('mid_interruptions'), 1)    # 2-5s
        self.assertEqual(timing_analysis.get('late_interruptions'), 1)   # >5s
        
        patterns = behavioral.get('interruption_patterns', {})
        self.assertEqual(patterns.get('max_consecutive_interruptions'), 3)
        self.assertEqual(patterns.get('interruption_consistency'), 1.0)  # All questions interrupted
    
    def test_result_message_generation(self):
        """Test generation of appropriate result messages"""
        # Test ADHD flagged message
        self.assessment.interruption_count = 4
        self.assessment.interruption_threshold = 3
        
        message = self.assessment._generate_assessment_message(True)
        self.assertIn("ADHD indicators detected", message)
        self.assertIn("4 interruptions recorded", message)
        self.assertIn("threshold: 3", message)
        self.assertIn("healthcare professional", message)
        
        # Test no ADHD message
        self.assessment.interruption_count = 1
        message = self.assessment._generate_assessment_message(False)
        self.assertIn("No ADHD indicators detected", message)
        self.assertIn("1 interruptions recorded", message)
    
    def test_disclaimer_generation(self):
        """Test generation of medical disclaimer - Requirements 3.4"""
        disclaimer = self.assessment._generate_disclaimer()
        
        self.assertIn("not diagnostic", disclaimer)
        self.assertIn("professional medical advice", disclaimer)
        self.assertIn("qualified healthcare professional", disclaimer)
        self.assertIn("medical professional", disclaimer)


class TestAssessmentStateManagement(unittest.TestCase):
    """Test cases for assessment state management and error handling"""
    
    def setUp(self):
        """Set up test fixtures for state management tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.assessment = InteractiveADHDAssessment(
            results_storage_path=os.path.join(self.temp_dir, "results"),
            session_storage_path=os.path.join(self.temp_dir, "sessions")
        )
    
    def tearDown(self):
        """Clean up after state management tests"""
        self.assessment.reset_assessment()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_get_current_state(self):
        """Test getting current assessment state"""
        state = self.assessment.get_current_state()
        
        self.assertEqual(state['state'], 'not_started')
        self.assertEqual(state['current_question_index'], 0)
        self.assertEqual(state['total_questions'], 5)
        self.assertEqual(state['interruption_count'], 0)
        self.assertFalse(state['is_presenting_question'])
        self.assertEqual(state['progress_percentage'], 0.0)
    
    def test_pause_and_resume_assessment(self):
        """Test pausing and resuming assessment"""
        # Start assessment first
        self.assessment.state = AssessmentState.IN_PROGRESS
        
        # Test pause
        result = self.assessment.pause_assessment()
        self.assertTrue(result)
        self.assertEqual(self.assessment.state, AssessmentState.PAUSED)
        
        # Test resume
        result = self.assessment.resume_assessment()
        self.assertTrue(result)
        self.assertEqual(self.assessment.state, AssessmentState.IN_PROGRESS)
    
    def test_pause_assessment_wrong_state(self):
        """Test pausing assessment when not in progress"""
        self.assessment.state = AssessmentState.NOT_STARTED
        
        result = self.assessment.pause_assessment()
        self.assertFalse(result)
        self.assertEqual(self.assessment.state, AssessmentState.NOT_STARTED)
    
    def test_reset_assessment(self):
        """Test complete assessment reset"""
        # Set up some state
        self.assessment.state = AssessmentState.IN_PROGRESS
        self.assessment.current_question_index = 2
        self.assessment.interruption_count = 3
        self.assessment.assessment_start_time = time.time()
        
        # Add a response
        response = QuestionResponse(
            question_index=0,
            question_text="Test",
            start_time=time.time(),
            completion_time=None,
            user_response=None,
            was_interrupted=False,
            interruption_time=None
        )
        self.assessment.responses.append(response)
        
        # Reset
        self.assessment.reset_assessment()
        
        # Verify reset
        self.assertEqual(self.assessment.state, AssessmentState.NOT_STARTED)
        self.assertEqual(self.assessment.current_question_index, 0)
        self.assertEqual(self.assessment.interruption_count, 0)
        self.assertEqual(len(self.assessment.responses), 0)
        self.assertIsNone(self.assessment.assessment_start_time)
        self.assertFalse(self.assessment.is_presenting_question)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestInteractiveADHDAssessment))
    test_suite.addTest(unittest.makeSuite(TestQuestionTimingAndInterruption))
    test_suite.addTest(unittest.makeSuite(TestResultCalculationLogic))
    test_suite.addTest(unittest.makeSuite(TestAssessmentStateManagement))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")