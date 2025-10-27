# Implementation Plan

- [x] 1. Create InteractiveADHDAssessment class





  - Create new file `interactive_assessment.py` with the main assessment orchestration class
  - Implement question storage, state management, and assessment flow control
  - Add methods for starting assessment, presenting questions, and calculating results
  - _Requirements: 1.1, 1.4, 1.5_

- [x] 2. Enhance existing ADHDDetector class





  - Modify `adhd_detection.py` to add question-based detection capabilities
  - Add methods for recording question interruptions and combining detection results
  - Integrate question-based tracking with existing audio-based detection
  - _Requirements: 1.1, 1.5_

- [x] 3. Implement question presentation and timing system





  - Add question management functionality to track presentation timing
  - Integrate with existing text-to-speech system for question delivery
  - Implement question completion detection and timeout handling
  - _Requirements: 1.1, 1.2, 5.3_

- [x] 4. Create response monitoring and interruption detection





  - Extend existing `ContinuousSpeechRecognizer` to detect interruptions during questions
  - Add timestamp recording for interruption events
  - Implement logic to differentiate between interruptions and valid responses
  - _Requirements: 1.3, 5.1, 5.2_

- [x] 5. Integrate assessment system with Gradio interface





  - Modify `main.py` to add assessment initiation button/command
  - Update chat interface to handle assessment mode
  - Ensure webcam feed remains active during assessment
  - Add visual indicators for assessment progress
  - _Requirements: 2.1, 2.2, 2.3, 4.1, 4.2_
-

- [x] 6. Implement result calculation and display system




  - Add logic to count interruptions and determine ADHD flagging
  - Create result display with appropriate messaging and disclaimers
  - Implement result storage and retrieval functionality
  - _Requirements: 1.5, 3.1, 3.2, 3.3, 3.4_

- [x] 7. Add assessment state management and error handling





  - Implement session state persistence for assessment progress
  - Add error handling for audio system failures and user disconnection
  - Ensure graceful recovery and continuation of assessment
  - _Requirements: 4.3, 4.4, 5.4_

- [x] 8. Create unit tests for assessment components






  - Write tests for InteractiveADHDAssessment class methods
  - Test question timing and interruption detection accuracy
  - Validate result calculation logic and edge cases
  - _Requirements: 1.1, 1.3, 1.5_

- [x] 9. Add integration tests for UI and audio pipeline






  - Test Gradio interface modifications and assessment flow
  - Verify webcam continuity during assessment
  - Test text-to-speech integration with question presentation
  - _Requirements: 2.1, 2.2, 4.1_