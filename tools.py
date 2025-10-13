import cv2
import base64
from dotenv import load_dotenv
import os
import platform

load_dotenv()

def capture_image() -> str:
    """
    Captures one frame from the default webcam, resizes it,
    encodes it as Base64 JPEG (raw string) and returns it.
    """
    # Determine the appropriate backend based on the operating system
    if platform.system() == "Windows":
        backend = cv2.CAP_DSHOW
    else:
        backend = cv2.CAP_AVFOUNDATION  # Or leave it as the default

    for idx in range(4):
        try:
            cap = cv2.VideoCapture(idx + backend)
            if not cap.isOpened():
                print(f"Could not open webcam with index {idx}")
                continue

            for _ in range(10):  # Warm up
                ret, _ = cap.read()
                if not ret:
                    print(f"Failed to read frame from webcam {idx} during warm-up")
                    break

            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame from webcam {idx}")
                cap.release()
                continue

            cv2.imwrite("sample.jpg", frame)  # Optional
            ret, buf = cv2.imencode('.jpg', frame)
            cap.release()  # Release the capture object here

            if ret:
                return base64.b64encode(buf).decode('utf-8')
            else:
                print(f"Failed to encode image from webcam {idx}")

        except Exception as e:
            print(f"An error occurred while processing webcam {idx}: {e}")

    raise RuntimeError("Could not open any webcam (tried indices 0-3)")

from groq import Groq

def analyze_image_with_query(query: str) -> str:
    """
    Expects a string with 'query'.
    Captures the image and sends the query and the image to
    to Groq's vision chat API and returns the analysis.
    """
    img_b64 = capture_image()
    model="meta-llama/llama-4-maverick-17b-128e-instruct"
    
    if not query or not img_b64:
        return "Error: both 'query' and 'image' fields required."

    client=Groq()  
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_b64}",
                    },
                },
            ],
        }]
    chat_completion=client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content

# query = "How many people do you see?"
# print(analyze_image_with_query(query))