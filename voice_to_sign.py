import speech_recognition as sr
import pygame
import sys

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        text = recognizer.recognize_google(audio)
        print(f"Recognized text: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I did not understand that.")
    except sr.RequestError:
        print("Could not request results; check your network connection.")

    return None

# Gesture mapping
gesture_map = {
    "hi": "gesture_data/Hi/1.png",
    "a": "gesture_data/A/1.png",
    # Add more mappings as needed
}

# Function to display gesture
def display_gesture(image_path):
    pygame.init()
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption("Sign Language Gesture")
    image = pygame.image.load(image_path)
    image = pygame.transform.scale(image, (640, 480))
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.blit(image, (0, 0))
        pygame.display.flip()

# Main function
def main():
    text = recognize_speech()
    if text:
        gesture_path = gesture_map.get(text.lower())
        if gesture_path:
            display_gesture(gesture_path)
        else:
            print("No gesture found for the recognized text.")

if __name__ == "__main__":
    main()
