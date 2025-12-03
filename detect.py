# detect.py
from ultralytics import YOLO
import pygame
import time
import threading

# Try to import RPi.GPIO; if not available (e.g. running on PC) we'll skip GPIO handling.
try:
    import RPi.GPIO as GPIO
    ON_RPI = True
except Exception:
    ON_RPI = False

# --- Optional GPIO Setup (only if running on Raspberry Pi with GPIO available) ---
BUTTON_PIN = 17
if ON_RPI:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# --- Load YOLO model ---
model = YOLO("runs/detect/train/weights/best.pt")

# --- Audio maps for each language ---
audio_map_en = {
    "Hello": "audio/en/Hello.mp3",
    "Yes": "audio/en/Yes.mp3",
    "No": "audio/en/No.mp3",
    "Thanks": "audio/en/Thanks.mp3",
    "IloveYou": "audio/en/IloveYou.mp3",
    "Please": "audio/en/Please.mp3",
}

audio_map_hi = {
    "Hello": "audio/hi/Hello.mp3",
    "Yes": "audio/hi/Yes.mp3",
    "No": "audio/hi/No.mp3",
    "Thanks": "audio/hi/Thanks.mp3",
    "IloveYou": "audio/hi/IloveYou.mp3",
    "Please": "audio/hi/Please.mp3",
}

audio_map_gu = {
    "Hello": "audio/gu/Hello.mp3",
    "Yes": "audio/gu/Yes.mp3",
    "No": "audio/gu/No.mp3",
    "Thanks": "audio/gu/Thanks.mp3",
    "IloveYou": "audio/gu/IloveYou.mp3",
    "Please": "audio/gu/Please.mp3",
}

# --- Language setup ---
languages = [audio_map_en, audio_map_hi, audio_map_gu]
lang_names = ["English", "Hindi", "Gujarati"]
current_lang_index = 0
audio_map = languages[current_lang_index]

# --- Init pygame (mixer + event system) ---
pygame.mixer.init()
pygame.init()

# Create a small window so pygame receives keyboard events.
# You can hide or minimize it but it must have focus to detect keys.
SCREEN_W, SCREEN_H = 200, 60
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("SignLang - Press SPACE to change language")
font = pygame.font.Font(None, 20)

def draw_status():
    screen.fill((30, 30, 30))
    text = f"Language: {lang_names[current_lang_index]}  (Press SPACE)"
    surf = font.render(text, True, (255,255,255))
    screen.blit(surf, (10, 20))
    pygame.display.flip()

draw_status()

last_spoken = None
last_time = 0
cooldown = 2  # seconds for audio repeat suppression

# Debounce for spacebar
last_key_time = 0
key_debounce = 0.5  # seconds

def play_audio(file):
    """Play audio in a background thread so detection loop is not blocked."""
    def _play():
        try:
            pygame.mixer.music.load(file)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"⚠️ Audio play error: {e} (file: {file})")
    threading.Thread(target=_play, daemon=True).start()

def cycle_language():
    """Cycle language index and update audio_map + on-screen label."""
    global current_lang_index, audio_map
    current_lang_index = (current_lang_index + 1) % len(languages)
    audio_map = languages[current_lang_index]
    print(f"🔄 Language changed to: {lang_names[current_lang_index]}")
    draw_status()

# --- If running on Raspberry Pi, setup GPIO callback too ---
if ON_RPI:
    def button_callback(channel):
        cycle_language()
    GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=button_callback, bouncetime=500)
    print("✅ GPIO button enabled (Raspberry Pi detected).")

print("✅ Sign Language Detection Started")
print(f"📷 Camera initialized. Current language: {lang_names[current_lang_index]}")
print("👉 Press SPACE to change language (English → Hindi → Gujarati → English...)")

# --- Run detection with webcam --- 
# model.predict returns a stream of 'result' objects when stream=True
for result in model.predict(source=0, conf=0.25, imgsz=256, show=True, stream=True):
    # Handle pygame events every iteration so keyboard input is responsive
    current_loop_time = time.time()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            # User closed pygame window; exit cleanly
            print("Exiting...")
            if ON_RPI:
                GPIO.cleanup()
            pygame.quit()
            raise SystemExit

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # debounce space key
                if (current_loop_time - last_key_time) > key_debounce:
                    last_key_time = current_loop_time
                    cycle_language()

    # If you prefer polling instead of event (keeps working if window has focus):
    # keys = pygame.key.get_pressed()
    # if keys[pygame.K_SPACE] and (time.time() - last_key_time) > key_debounce:
    #     last_key_time = time.time()
    #     cycle_language()

    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        # class ids from detection
        class_ids = boxes.cls.cpu().numpy().astype(int)
        for cls in class_ids:
            label = model.names[cls]
            current_time = time.time()
            if label in audio_map and (label != last_spoken or (current_time - last_time) > cooldown):
                print(f"✅ Detected: {label} → Playing in {lang_names[current_lang_index]}")
                play_audio(audio_map[label])
                last_spoken = label
                last_time = current_time
