import cv2
from get_model import get_model
from torchvision import transforms
import torch
from PIL import Image
import pathlib


def ccaapp(file_path):
    return cv2.VideoCapture(file_path)

def big(cap):
    test_transforms = transforms.Compose([transforms.ToTensor()])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = get_model()

    crop = (260, 550, 1300, 1080)

    threshold = 0.2


    while True:
        _, frame = cap.read()
        if frame is None:
            break

        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = test_transforms(im_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(inputs)[0]

        for box, score in zip(list(outputs["boxes"]), list(outputs["scores"])):
            x_min, y_min, x_max, y_max = box
            if score > threshold:
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

        color = (119, 201, 105)
        cv2.line(frame, (330, 750), (290, 800), color, thickness=3)  # левая вертикальная
        cv2.line(frame, (290, 800), (1270, 800), color, thickness=3)  # Нижняя горизонтальная
        cv2.line(frame, (330, 750), (1240, 750), color, thickness=3)  # Верхняя горизонтальная
        cv2.line(frame, (1240, 750), (1270, 800), color, thickness=3)  # Правая вертикальная

        cv2.rectangle(frame, (crop[0], crop[1]), (crop[2], crop[3]), (0, 0, 255), 2)

        cv2.namedWindow("video", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("video", cv2.WND_PROP_FULLSCREEN, 1)
        cv2.imshow("video", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
