import cv2
import torch
from model import DETR
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler
import pyttsx3
import time

# -------------------------------------------------
# WORD DICTIONARY & WORD BREAK
# -------------------------------------------------
WORD_DICTIONARY = {
    "i", "love", "you",
    "thank",
    "hello",
    "yes",
    "no",
    "food",
    "sorry"
}

def word_break(s, word_dict):
    s = s.lower()
    memo = {}

    def dfs(i):
        if i == len(s):
            return [""]
        if i in memo:
            return memo[i]

        res = []
        for j in range(i + 1, len(s) + 1):
            word = s[i:j]
            if word in word_dict:
                for rest in dfs(j):
                    res.append(word + ("" if rest == "" else " " + rest))
        memo[i] = res
        return res

    out = dfs(0)
    return out[0] if out else s

# -------------------------------------------------
# LOGGER & SETUP
# -------------------------------------------------
logger = get_logger("realtime")
detection_handler = DetectionHandler()
logger.print_banner()
logger.realtime("Starting real-time sign language detection")

# -------------------------------------------------
# TRANSFORMS
# -------------------------------------------------
transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# -------------------------------------------------
# MODEL (7 CLASSES)
# -------------------------------------------------
model = DETR(num_classes=7)
model.eval()
model.load_pretrained("checkpoints/99_model.pt")

CLASSES = get_classes()   # must return 7 classes
COLORS = get_colors()     # must return 7 colors

# -------------------------------------------------
# CAMERA
# -------------------------------------------------
cap = cv2.VideoCapture(0)
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)

detected_words = set()
fps_start = time.time()
frame_count = 0

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    transformed = transforms(image=frame)
    image_tensor = transformed["image"].unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

    probs = output["pred_logits"].softmax(-1)[:, :, :-1]
    max_probs, max_classes = probs.max(-1)
    keep = max_probs > 0.8

    batch_idx, query_idx = torch.where(keep)
    boxes = rescale_bboxes(
        output["pred_boxes"][batch_idx, query_idx],
        frame.shape[:2][::-1]
    )

    for cls, prob, box in zip(
        max_classes[batch_idx, query_idx],
        max_probs[batch_idx, query_idx],
        boxes
    ):
        idx = int(cls)
        label = CLASSES[idx]
        detected_words.add(label)

        x1, y1, x2, y2 = map(int, box)
        color = COLORS[idx]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            frame,
            f"{label} {prob:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    frame_count += 1
    if frame_count % 30 == 0:
        fps = 30 / (time.time() - fps_start)
        fps_start = time.time()
        detection_handler.log_inference_time(0, fps)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------------------------
# TEXT TO SPEECH (AFTER EXIT)
# -------------------------------------------------
if detected_words:
    sentence = ""
    for w in detected_words:
        sentence += word_break(w, WORD_DICTIONARY) + " "

    sentence = sentence.strip()
    print("Detected sentence:", sentence)

    engine = pyttsx3.init()
    engine.setProperty("rate", 110)
    engine.say(sentence)
    engine.runAndWait()
else:
    print("No signs detected.")
