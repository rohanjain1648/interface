# Dora - Personal AI Assistant with ADHD Detection

Dora is an interactive AI assistant featuring live webcam integration, continuous speech recognition, and experimental ADHD detection capabilities through behavioral pattern analysis.

## ⚠️ Important Disclaimer

This application includes experimental ADHD detection features. **Results are not diagnostic and should not replace professional medical advice.** Always consult a healthcare professional for any medical concerns.

## Features

- 🎥 **Live Webcam Feed** - Real-time video streaming with optimized performance
- 🎤 **Continuous Speech Recognition** - Always-listening voice interaction
- 🤖 **AI Chat Assistant** - Powered by Groq API for intelligent conversations
- 🔊 **Text-to-Speech** - Audio responses using ElevenLabs or Google TTS
- 🧠 **ADHD Detection** - Experimental behavioral pattern analysis
- 📊 **Interactive Assessment** - 5-question assessment system for interruption pattern detection

## Installation

### Prerequisites

- Python 3.8+
- Microphone and webcam access
- Internet connection for AI services

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd dora
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

## Configuration

### Required API Keys

- **Groq API**: For AI chat functionality
  - Get your key from [Groq Console](https://console.groq.com/)
- **ElevenLabs API**: For high-quality text-to-speech (optional)
  - Get your key from [ElevenLabs](https://elevenlabs.io/)
  - Falls back to Google TTS if not configured

### Audio Settings

The application uses the following audio settings:
- Sample rate: 16kHz
- Frame duration: 30ms
- Voice Activity Detection: WebRTC VAD (aggressiveness level 3)

## Usage

1. **Start the Application**
   - Run `python main.py`
   - Open your browser to the displayed URL (typically `http://localhost:7860`)

2. **Basic Chat**
   - Click "Start Camera" to enable webcam
   - Speak naturally - the system is always listening
   - View responses in the chat interface and hear audio replies

3. **ADHD Assessment**
   - Click "Start ADHD Assessment" to begin the 5-question evaluation
   - Listen to each question completely before responding
   - The system tracks interruption patterns during question presentation
   - Results are displayed after completing all 5 questions

## Project Structure

```
dora/
├── main.py                 # Main Gradio application
├── adhd_detection.py       # ADHD detection algorithms
├── interactive_assessment.py # Question-based assessment system
├── speech_to_text.py       # Speech recognition functionality
├── text_to_speech.py       # Text-to-speech functionality
├── ai_agent.py            # AI chat agent
├── tools.py               # Utility functions
├── .kiro/                 # Kiro IDE specifications
│   └── specs/
│       └── interactive-adhd-detection/
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Development

### Specifications

This project uses Kiro IDE specifications for structured development. The interactive ADHD detection feature is fully specified in `.kiro/specs/interactive-adhd-detection/`:

- `requirements.md` - Feature requirements and acceptance criteria
- `design.md` - System architecture and component design
- `tasks.md` - Implementation tasks and development plan

### Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the existing code style
4. Add tests for new functionality
5. Submit a pull request

## Technical Details

### ADHD Detection Methods

1. **Audio-based Detection**
   - Monitors speech patterns and interruption frequency
   - Uses sliding window analysis (default: 5 interactions)
   - Flags potential ADHD indicators based on interruption threshold

2. **Question-based Assessment**
   - Presents 5 structured questions about daily routines and focus
   - Tracks interruptions during question presentation
   - Flags ADHD indicators if 3+ interruptions occur

### Performance Optimizations

- **Webcam**: 30 FPS with buffer optimization to reduce lag
- **Audio**: Non-blocking processing with threading
- **Speech Recognition**: Continuous listening with VAD filtering
- **UI**: Responsive design with real-time updates

## Troubleshooting

### Common Issues

1. **Microphone not working**
   - Check system permissions for microphone access
   - Ensure no other applications are using the microphone

2. **Webcam not starting**
   - Verify webcam permissions in browser/system settings
   - Try refreshing the page or restarting the application

3. **Audio playback issues**
   - Check system audio settings
   - Verify API keys are correctly configured

4. **High CPU usage**
   - Reduce webcam frame rate in settings
   - Close other resource-intensive applications

## License

This project is for educational and research purposes. Please ensure compliance with all applicable laws and regulations when using ADHD detection features.

---

**Remember**: This tool is experimental and not a substitute for professional medical diagnosis or treatment.
