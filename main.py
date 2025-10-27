import os
import gradio as gr
from speech_to_text import record_audio, transcribe_with_groq, ContinuousSpeechRecognizer
from ai_agent import ask_agent
from text_to_speech import text_to_speech_with_elevenlabs, text_to_speech_with_gtts
from adhd_detection import ADHDDetector
from interactive_assessment import InteractiveADHDAssessment, AssessmentState
from dotenv import load_dotenv  # Import load_dotenv
import subprocess
import platform
import logging
import threading
import time

load_dotenv()  # Load environment variables from .env file
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
audio_filepath = "audio_question.mp3"
output_audio_filepath = "final.wav"  # Changed to WAV for broader compatibility

# Initialize ADHD detector and assessment system
adhd_detector = ADHDDetector()
speech_recognizer = ContinuousSpeechRecognizer(adhd_detector=adhd_detector)
assessment_system = InteractiveADHDAssessment()

# Global assessment state
assessment_active = False
assessment_results = None

logging.basicConfig(level=logging.INFO)

def play_audio(filepath):
    """Plays the audio file using the appropriate command for the OS."""
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', filepath], check=True)
        elif os_name == "Windows":  # Windows
            try:
                from playsound import playsound
                playsound(filepath)
            except ImportError:
                print("playsound not installed. Please install it with 'pip install playsound'")
            except Exception as e:
                print(f"Error playing audio with playsound: {e}")
        elif os_name == "Linux":  # Linux
            subprocess.run(['aplay', filepath], check=True)  # Alternative: use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported operating system")
    except subprocess.CalledProcessError as e:
        print(f"Error playing audio: {e}")
    except OSError as e:
        print(f"Unsupported operating system: {e}")

def start_adhd_assessment():
    """Start the ADHD assessment process with error handling and recovery"""
    global assessment_active, assessment_results
    
    if assessment_active:
        return "Assessment already in progress", None, "Assessment in progress..."
    
    try:
        # Check for existing session to recover
        existing_sessions = assessment_system.list_available_sessions()
        if existing_sessions:
            latest_session = existing_sessions[0]
            if latest_session["state"] in ["in_progress", "paused"]:
                logging.info(f"Found recoverable session: {latest_session['session_id']}")
                if assessment_system.load_session_state(latest_session["session_id"]):
                    assessment_active = True
                    return (
                        [["System", f"Recovered assessment session. Continuing from question {latest_session['current_question']}/5"]],
                        "assessment_audio.wav",
                        f"Assessment recovered. Question {latest_session['current_question']}/5 - Ready to continue."
                    )
        
        # Reset assessment system for fresh start
        assessment_system.reset_assessment()
        assessment_results = None
        
        # Set up enhanced assessment callbacks with error handling
        def on_question_start(question_index, question_text):
            logging.info(f"Assessment question {question_index + 1} started: {question_text[:50]}...")
        
        def on_question_complete(question_index):
            logging.info(f"Assessment question {question_index + 1} presentation completed")
        
        def on_interruption(count, timestamp):
            logging.info(f"Assessment interruption detected. Total: {count}")
        
        def on_assessment_complete(results):
            global assessment_active, assessment_results
            assessment_active = False
            assessment_results = results
            logging.info("Assessment completed")
        
        def on_error(error_type, error_data):
            logging.warning(f"Assessment error: {error_type} - {error_data}")
            if error_type == "user_disconnection":
                logging.info("User disconnection detected - session preserved for recovery")
        
        def on_recovery(recovery_type, recovery_data):
            logging.info(f"Assessment recovery: {recovery_type} - {recovery_data}")
        
        # Configure assessment system callbacks with error handling
        assessment_system.set_callbacks(
            on_question_start=on_question_start,
            on_question_complete=on_question_complete,
            on_interruption=on_interruption,
            on_assessment_complete=on_assessment_complete
        )
        
        # Set error and recovery callbacks
        assessment_system.on_error_callback = on_error
        assessment_system.on_recovery_callback = on_recovery
        
        # Re-setup speech integration to ensure callbacks are current
        setup_assessment_speech_integration()
        
        # Start assessment with error handling
        if assessment_system.start_assessment():
            assessment_active = True
            state = assessment_system.get_current_state()
            current_question = assessment_system.ASSESSMENT_QUESTIONS[0][:100] + "..."
            return (
                [["System", f"ADHD Assessment Started. Question 1: {current_question}"]],
                "assessment_audio.wav",
                "Assessment started. Question 1/5 - Listen carefully and respond naturally."
            )
        else:
            error_summary = assessment_system.get_error_summary()
            error_msg = f"Failed to start assessment. Errors: {error_summary['total_errors']}"
            return error_msg, None, "Error starting assessment"
            
    except Exception as e:
        logging.error(f"Error starting assessment: {e}")
        return f"Error: {str(e)}", None, "Error starting assessment"

def get_assessment_progress():
    """Get current assessment progress with detailed status"""
    if not assessment_active:
        if assessment_results:
            # Use the enhanced result display formatting
            ui_results = assessment_system.get_results_for_ui_display(assessment_results)
            return f"✅ Assessment Complete - {ui_results['main_message']}"
        return "No assessment in progress"
    
    state = assessment_system.get_current_state()
    progress = state['progress_percentage']
    current_q = state['current_question_index'] + 1
    total_q = state['total_questions']
    interruptions = state['interruption_count']
    
    # Add presentation status
    if assessment_system.is_presenting_question:
        status_icon = "🎤"
        status_text = "Presenting question"
    else:
        status_icon = "⏳"
        status_text = "Waiting for response"
    
    return f"{status_icon} Question {current_q}/{total_q} ({progress:.0f}%) - {status_text} - Interruptions: {interruptions}"

def get_assessment_results_display():
    """Get formatted assessment results for display"""
    global assessment_results
    
    if assessment_results:
        return assessment_system.format_results_for_display(assessment_results)
    elif assessment_system.stored_results:
        return assessment_system.format_results_for_display(assessment_system.stored_results)
    else:
        latest_results = assessment_system.get_latest_results()
        if latest_results:
            return assessment_system.format_results_for_display(latest_results)
    
    return "No assessment results available."

def get_assessment_results_summary():
    """Get brief assessment results summary"""
    global assessment_results
    
    if assessment_results:
        return assessment_system.format_results_summary(assessment_results)
    elif assessment_system.stored_results:
        return assessment_system.format_results_summary(assessment_system.stored_results)
    else:
        latest_results = assessment_system.get_latest_results()
        if latest_results:
            return assessment_system.format_results_summary(latest_results)
    
    return "No assessment results available."

def recover_assessment():
    """Attempt to recover from assessment errors"""
    global assessment_active, assessment_results
    
    try:
        if assessment_system.state == AssessmentState.ERROR:
            logging.info("Attempting assessment recovery from error state")
            
            if assessment_system.recover_from_error():
                assessment_active = True
                state = assessment_system.get_current_state()
                current_q = state['current_question_index'] + 1
                
                return (
                    [["System", f"Assessment recovered successfully. Continuing from question {current_q}/5"]],
                    "assessment_audio.wav",
                    f"Recovery successful. Question {current_q}/5 - Ready to continue."
                )
            else:
                error_summary = assessment_system.get_error_summary()
                return (
                    [["System", f"Recovery failed. Total errors: {error_summary['total_errors']}"]],
                    None,
                    "Recovery failed. Please restart assessment."
                )
        else:
            return (
                [["System", "No recovery needed - assessment not in error state"]],
                None,
                f"Assessment state: {assessment_system.state.value}"
            )
            
    except Exception as e:
        logging.error(f"Error during recovery attempt: {e}")
        return (
            [["System", f"Recovery error: {str(e)}"]],
            None,
            "Recovery attempt failed"
        )

def ensure_webcam_during_assessment():
    """Ensure webcam remains active during assessment"""
    # This function ensures webcam continues running during assessment
    # The webcam timer in the UI will keep calling get_webcam_frame()
    # which maintains the feed regardless of assessment state
    return get_webcam_frame()

def setup_assessment_speech_integration():
    """Set up integration between speech recognition and assessment system"""
    
    def on_assessment_interruption(transcription, timestamp):
        """Handle interruption detected during assessment"""
        logging.info(f"Assessment interruption: {transcription}")
        assessment_system.record_interruption(timestamp)
    
    def on_assessment_response(transcription, timestamp):
        """Handle valid response during assessment"""
        logging.info(f"Assessment response: {transcription}")
        assessment_system.handle_user_response(transcription, timestamp)
    
    # Set up speech recognizer callbacks for assessment
    def start_question_monitoring():
        """Start monitoring for the current question"""
        speech_recognizer.start_question_monitoring(
            interruption_callback=on_assessment_interruption,
            response_callback=on_assessment_response
        )
    
    def mark_question_complete():
        """Mark current question as complete"""
        speech_recognizer.mark_question_completion()
    
    def stop_question_monitoring():
        """Stop question monitoring"""
        speech_recognizer.stop_question_monitoring()
    
    # Set up assessment callbacks to control speech monitoring
    def on_question_start(question_index, question_text):
        """Called when assessment question starts"""
        start_question_monitoring()
        logging.info(f"Started monitoring for question {question_index + 1}")
    
    def on_question_complete(question_index):
        """Called when assessment question completes"""
        mark_question_complete()
        logging.info(f"Marked question {question_index + 1} as complete")
    
    def on_assessment_complete(results):
        """Called when assessment completes"""
        stop_question_monitoring()
        logging.info("Stopped question monitoring - assessment complete")
    
    # Configure assessment system with speech integration callbacks
    assessment_system.set_callbacks(
        on_question_start=on_question_start,
        on_question_complete=on_question_complete,
        on_assessment_complete=on_assessment_complete
    )

def process_audio_and_chat():
    chat_history = []
    adhd_detector.start_audio_stream()  # Start ADHD detection audio stream
    
    # Set up assessment-speech integration
    setup_assessment_speech_integration()

    def on_speech_detected(transcription):
        nonlocal chat_history
        try:
            user_input = transcription

            if "goodbye" in user_input.lower():
                speech_recognizer.stop_continuous_listening()
                adhd_detector.stop_audio_stream()
                return

            # Handle assessment mode
            if assessment_active:
                # Speech is handled by the assessment system through callbacks
                # Just update chat with assessment status
                if assessment_system.state == AssessmentState.COMPLETED:
                    results = assessment_system.calculate_results()
                    ui_results = assessment_system.get_results_for_ui_display(results)
                    chat_history.append([user_input, f"Assessment completed. {ui_results['main_message']}"])
                else:
                    state = assessment_system.get_current_state()
                    current_q = state['current_question_index'] + 1
                    total_q = state['total_questions']
                    if assessment_system.is_presenting_question:
                        chat_history.append([user_input, f"Response during question {current_q}/{total_q} (interruption detected)"])
                    else:
                        chat_history.append([user_input, f"Response recorded for question {current_q}/{total_q}"])
            else:
                # Normal chat mode
                response = ask_agent(user_query=user_input)

                text_to_speech_with_elevenlabs(input_text=response, output_filepath=output_audio_filepath)

                # Check for ADHD after each interaction
                adhd_status = adhd_detector.get_adhd_status()
                if adhd_status["flagged"]:
                    logging.warning("ADHD-like behavior detected based on interruption patterns.")
                    # Optionally, add to response or log
                    response += " (Note: High interruption rate detected - consider consulting a professional for ADHD assessment.)"

                chat_history.append([user_input, response])

            # Yield updated history and audio
            # Note: In Gradio, yielding from a callback might not work as expected; adjust as needed

        except Exception as e:
            logging.error(f"Error processing speech: {e}")

    speech_recognizer.start_continuous_listening(on_speech_detected)

    # Initial yield
    yield chat_history, None

    # Keep yielding updates (this is simplified; real-time updates may need adjustment)
    while speech_recognizer.is_listening:
        yield chat_history, output_audio_filepath
        import time
        time.sleep(1)  # Poll for updates

# Code for frontend
import cv2
# Global variables
camera = None
is_running = False
last_frame = None

def initialize_camera():
    """Initialize the camera with optimized settings"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if camera.isOpened():
            # Optimize camera settings for better performance
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
    return camera is not None and camera.isOpened()

def start_webcam():
    """Start the webcam feed"""
    global is_running, last_frame
    is_running = True
    if not initialize_camera():
        return None
    
    ret, frame = camera.read()
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        return frame
    return last_frame

def stop_webcam():
    """Stop the webcam feed"""
    global is_running, camera
    is_running = False
    if camera is not None:
        camera.release()
        camera = None
    return None

def get_webcam_frame():
    """Get current webcam frame with optimized performance"""
    global camera, is_running, last_frame
    
    if not is_running or camera is None:
        return last_frame
    
    # Skip frames if buffer is full to avoid lag
    if camera.get(cv2.CAP_PROP_BUFFERSIZE) > 1:
        for _ in range(int(camera.get(cv2.CAP_PROP_BUFFERSIZE)) - 1):
            camera.read()
    
    ret, frame = camera.read()
    if ret and frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        last_frame = frame
        return frame
    return last_frame

# Setup UI

with gr.Blocks() as demo:
    gr.Markdown("<h1 style='color: orange; text-align: center;  font-size: 4em;'> 👧🏼 Dora – Your Personal AI Assistant</h1>")
    gr.Markdown("<p style='text-align: center; color: red; font-weight: bold;'>⚠️ WARNING: This app includes experimental ADHD detection features. Results are not diagnostic and should not replace professional medical advice. Consult a healthcare professional for any concerns.</p>")

    with gr.Row():
        # Left column - Webcam
        with gr.Column(scale=1):
            gr.Markdown("## Webcam Feed")
            
            with gr.Row():
                start_btn = gr.Button("Start Camera", variant="primary")
                stop_btn = gr.Button("Stop Camera", variant="secondary")
            
            webcam_output = gr.Image(
                label="Live Feed",
                streaming=True,
                show_label=False,
                width=640,
                height=480
            )
            
            # Faster refresh rate for smoother video
            webcam_timer = gr.Timer(0.033)  # ~30 FPS (1/30 ≈ 0.033 seconds)
        
        # Right column - Chat
        with gr.Column(scale=1):
            gr.Markdown("## Chat Interface")
            
            chatbot = gr.Chatbot(
                label="Conversation",
                height=300,
                show_label=False
            )
            
            audio_output = gr.Audio(label="Audio Response") # Add audio component
            
            # Assessment section
            gr.Markdown("### ADHD Assessment")
            gr.Markdown("*This assessment monitors response patterns during 5 questions. Results are not diagnostic.*")
            
            with gr.Row():
                start_assessment_btn = gr.Button("Start ADHD Assessment", variant="primary")
                recover_assessment_btn = gr.Button("Recover Assessment", variant="secondary")
                assessment_progress = gr.Textbox(
                    label="Assessment Status",
                    value="No assessment in progress",
                    interactive=False,
                    max_lines=3
                )
            
            # Assessment Results Display
            with gr.Accordion("Assessment Results", open=False) as results_accordion:
                results_summary = gr.Textbox(
                    label="Results Summary",
                    value="No results available",
                    interactive=False,
                    max_lines=2
                )
                
                results_display = gr.Textbox(
                    label="Detailed Results",
                    value="No results available",
                    interactive=False,
                    max_lines=20,
                    show_label=False
                )
                
                with gr.Row():
                    refresh_results_btn = gr.Button("Refresh Results", variant="secondary")
                    view_stored_btn = gr.Button("View Stored Assessments", variant="secondary")
            
            gr.Markdown("*🎤 Continuous listening mode is active - speak anytime!*")
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")
    
    # Event handlers
    start_btn.click(
        fn=start_webcam,
        outputs=webcam_output
    )
    
    stop_btn.click(
        fn=stop_webcam,
        outputs=webcam_output
    )
    
    webcam_timer.tick(
        fn=get_webcam_frame,
        outputs=webcam_output,
        show_progress=False  # Hide progress indicator for smoother experience
    )
    
    clear_btn.click(
        fn=lambda: [],
        outputs=chatbot
    )
    
    # Assessment event handlers
    start_assessment_btn.click(
        fn=start_adhd_assessment,
        outputs=[chatbot, audio_output, assessment_progress]
    )
    
    recover_assessment_btn.click(
        fn=recover_assessment,
        outputs=[chatbot, audio_output, assessment_progress]
    )
    
    # Update assessment progress periodically
    assessment_timer = gr.Timer(2.0)  # Update every 2 seconds
    assessment_timer.tick(
        fn=get_assessment_progress,
        outputs=assessment_progress,
        show_progress=False
    )
    
    # Result display event handlers
    refresh_results_btn.click(
        fn=lambda: (get_assessment_results_summary(), get_assessment_results_display()),
        outputs=[results_summary, results_display]
    )
    
    def show_stored_assessments():
        """Show list of stored assessments"""
        assessments = assessment_system.list_stored_assessments()
        if not assessments:
            return "No stored assessments found.", "No stored assessments found."
        
        # Create summary of stored assessments
        summary_lines = [f"Found {len(assessments)} stored assessments:"]
        for i, assessment in enumerate(assessments[:5]):  # Show latest 5
            status = "✅" if assessment.get("assessment_completed", False) else "⚠️"
            adhd_flag = "🔴" if assessment.get("adhd_flagged", False) else "🟢"
            date = assessment.get("datetime", "Unknown date")[:19] if assessment.get("datetime") else "Unknown date"
            interruptions = assessment.get("interruption_count", 0)
            
            summary_lines.append(f"{i+1}. {status} {adhd_flag} {date} - {interruptions} interruptions")
        
        if len(assessments) > 5:
            summary_lines.append(f"... and {len(assessments) - 5} more assessments")
        
        # Show details of the latest assessment
        latest = assessments[0]
        latest_results = assessment_system.retrieve_results(latest["assessment_id"])
        if latest_results:
            detailed_display = assessment_system.format_results_for_display(latest_results)
        else:
            detailed_display = "Could not load latest assessment details."
        
        return "\n".join(summary_lines), detailed_display
    
    view_stored_btn.click(
        fn=show_stored_assessments,
        outputs=[results_summary, results_display]
    )
    
    # Auto-start continuous mode when the app loads
    demo.load(
        fn=process_audio_and_chat,
        outputs=[chatbot, audio_output] # Return both chatbot and audio
    )

## Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )