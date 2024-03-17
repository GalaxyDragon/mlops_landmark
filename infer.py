from implementations.config import VALID_TRANSFORM, MODEL_PATH
import torch
import cv2

if __name__ == '__main__':
    model = torch.load(MODEL_PATH+'/model.pkl')
    img_path = './infer.jpg'

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    image = [VALID_TRANSFORM(image=img)["image"]]
    outputs = model(image)
    _, preds = torch.max(outputs, 1)
    pred = preds[0]
    print(pred)
