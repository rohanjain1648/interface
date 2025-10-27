# Requirements Document

## Introduction

This feature enhances the existing Dora AI assistant with an interactive ADHD detection system. The system will present users with 5 questions and monitor their response behavior to detect potential ADHD indicators. If a user interrupts or responds before question completion 3 or more times, the system will flag potential ADHD behavior and display an appropriate notification.

## Glossary

- **Dora_System**: The existing AI assistant application with webcam and chat capabilities
- **Question_Module**: The component responsible for presenting ADHD assessment questions
- **Response_Monitor**: The component that tracks user interruption behavior during questions
- **ADHD_Detector**: The enhanced detection system that combines audio-based and question-based detection
- **Interruption_Event**: When a user speaks or responds before a question is fully presented
- **Assessment_Session**: A complete cycle of 5 questions presented to the user
- **Flag_Threshold**: The minimum number of interruptions (3) required to trigger ADHD detection

## Requirements

### Requirement 1

**User Story:** As a user of Dora, I want to participate in an interactive ADHD assessment, so that I can receive feedback about potential ADHD indicators based on my response patterns.

#### Acceptance Criteria

1. WHEN the user initiates an ADHD assessment, THE Dora_System SHALL present exactly 5 questions sequentially
2. WHILE each question is being presented, THE Response_Monitor SHALL track any interruption events
3. IF the user speaks before question completion, THEN THE Response_Monitor SHALL record an interruption event
4. WHERE the user completes all 5 questions, THE Dora_System SHALL calculate the total interruption count
5. WHEN interruption count reaches 3 or more, THE ADHD_Detector SHALL flag potential ADHD behavior

### Requirement 2

**User Story:** As a user, I want the live webcam interface to remain interactive throughout the ADHD assessment, so that I can maintain visual engagement with the system.

#### Acceptance Criteria

1. WHILE the Assessment_Session is active, THE Dora_System SHALL maintain continuous webcam feed display
2. THE Dora_System SHALL ensure webcam frame updates occur at minimum 15 FPS during assessment
3. WHEN questions are being presented, THE Dora_System SHALL keep the webcam interface responsive
4. THE Dora_System SHALL display both webcam feed and question interface simultaneously

### Requirement 3

**User Story:** As a user, I want to receive clear feedback about my assessment results, so that I understand whether potential ADHD indicators were detected.

#### Acceptance Criteria

1. WHEN interruption count is less than 3, THE Dora_System SHALL display "No ADHD indicators detected"
2. WHEN interruption count equals or exceeds 3, THE Dora_System SHALL display "ADHD indicators detected - consider professional consultation"
3. THE Dora_System SHALL show the total interruption count to the user
4. THE Dora_System SHALL include appropriate disclaimers about the assessment being non-diagnostic

### Requirement 4

**User Story:** As a user, I want the assessment to integrate seamlessly with the existing chat interface, so that I can access it naturally within the current application flow.

#### Acceptance Criteria

1. THE Dora_System SHALL provide a clear method to initiate the ADHD assessment from the chat interface
2. WHEN assessment is active, THE Dora_System SHALL temporarily modify the chat behavior to focus on questions
3. WHEN assessment completes, THE Dora_System SHALL return to normal chat functionality
4. THE Dora_System SHALL preserve chat history before and after assessment sessions

### Requirement 5

**User Story:** As a user, I want the system to handle my responses appropriately during the assessment, so that my answers are recorded even if I interrupt questions.

#### Acceptance Criteria

1. WHEN the user interrupts a question, THE Response_Monitor SHALL record both the interruption and the response content
2. THE Question_Module SHALL continue to the next question after recording the user's response
3. IF the user provides no response within 10 seconds of question completion, THE Question_Module SHALL proceed to the next question
4. THE Dora_System SHALL ensure all 5 questions are presented regardless of interruption patterns