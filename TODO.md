# TODO: Add ADHD Detection Feature

## Steps to Complete

- [x] Update pyproject.toml to add new dependencies: webrtcvad, speechbrain, pyaudio
- [x] Create adhd_detection.py module with VAD, speaker diarization, interruption detection, and ADHD logic (sliding window)
- [x] Modify speech_to_text.py to use continuous audio streaming with pyaudio for real-time processing
- [x] Update main.py to integrate ADHD detection in the conversation loop, track interruptions, and log ADHD indicators
- [x] Add ethical disclaimers and consent warnings in the UI
- [x] Install new dependencies
- [x] Test real-time audio processing and interruption detection
- [ ] Final testing and refinements
