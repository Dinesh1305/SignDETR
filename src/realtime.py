import cv2
import torch
from torch import load
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
# SETUP
# -------------------------------------------------
logger = get_logger("realtime")
detection_handler = DetectionHandler()
logger.print_banner()
logger.realtime("Initializing real-time sign language detection...")

transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

model = DETR(num_classes=3)
model.eval()
model.load_pretrained('checkpoints/99_model.pt')

CLASSES = get_classes()
COLORS = get_colors()

logger.realtime("Starting camera capture...")
cap = cv2.VideoCapture(0)

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

# Set for storing detected unique words
detected_words = set()

frame_count = 0
fps_start_time = time.time()

# -------------------------------------------------
# MAIN LOOP
# -------------------------------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to read frame from camera")
        break

    # MODEL INFERENCE
    inference_start = time.time()
    transformed = transforms(image=frame)
    result = model(torch.unsqueeze(transformed['image'], dim=0))
    inference_time = (time.time() - inference_start) * 1000

    probabilities = result['pred_logits'].softmax(-1)[:, :, :-1]
    max_probs, max_classes = probabilities.max(-1)
    keep_mask = max_probs > 0.8

    batch_indices, query_indices = torch.where(keep_mask)
    bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices, :], (1920, 1080))
    classes = max_classes[batch_indices, query_indices]
    probas = max_probs[batch_indices, query_indices]

    detections = []

    for bclass, bprob, bbox in zip(classes, probas, bboxes):
        bclass_idx = bclass.detach().numpy()
        bprob_val = bprob.detach().numpy()
        x1, y1, x2, y2 = bbox.detach().numpy()

        detections.append({
            'class': CLASSES[bclass_idx],
            'confidence': float(bprob_val),
            'bbox': [float(x1), float(y1), float(x2), float(y2)]
        })

        # STORE DETECTED WORD
        detected_words.add(CLASSES[bclass_idx])

        # DRAW BOX
        frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              COLORS[bclass_idx], 8)
        label = f"{CLASSES[bclass_idx]} - {round(float(bprob_val), 3)}"

        cv2.rectangle(frame, (int(x1), int(y1) - 60), (int(x1) + 300, int(y1)),
                      COLORS[bclass_idx], -1)
        cv2.putText(frame, label, (int(x1), int(y1) - 15),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    # FPS COUNTER
    frame_count += 1
    if frame_count % 30 == 0:
        elapsed_time = time.time() - fps_start_time
        fps = 30 / elapsed_time
        if detections:
            detection_handler.log_detections(detections, frame_id=frame_count)
            detection_handler.log_inference_time(inference_time, fps)
        fps_start_time = time.time()

    # DISPLAY
    frame = cv2.resize(frame, (1920, 1080))
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.realtime("Stopping real-time detection...")
        break

cap.release()

# -------------------------------------------------
# SPEAK DETECTED WORDS AFTER WINDOW CLOSES
# -------------------------------------------------
print("\nDetected Sign Words (Unique):")
if detected_words:
    for w in detected_words:
        print(w)

    engine = pyttsx3.init()
    engine.setProperty('rate', 110)  # slow readable speed
    engine.setProperty('volume', 3.0)

    print("\nDetected Sign Words (Unique):")
    y=""
    for word in detected_words:
        y+=word  # print nicely
        y+=" "
    print(y.strip())

    # Read the whole sentence slowly
    engine.say(y)

    engine.runAndWait()
else:
    print("No words detected during the session.")
print(detected_words)
cv2.destroyAllWindows()
