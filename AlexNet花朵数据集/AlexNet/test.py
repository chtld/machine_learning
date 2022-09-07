import os
import json
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device".format(device))

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    base_path = "C:\\Users\\zhangshangyuan\\Desktop\\machine_learning\\AlexNet花朵数据集\\"
    # load_image
    img_path = base_path + "flower\\val\\roses\\1402130395_0b89d76029.jpg"
    assert os.path.exists(img_path), "file '{}' does not exit".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)

    json_path = base_path + "flower\\class_indices.json"
    assert os.path.exists(
        json_path), "file '{}' does not exit".format(json_path)
    json_file = open(json_path, "r")
    # {'0': 'daisy', '1': 'dandelion', '2': 'roses', '3': 'sunflowers', '4': 'tulips'}
    class_index = json.load(json_file)

    # create model and load parameters
    model = AlexNet(num_class=5).to(device)
    weights_path = base_path + 'AlexNet\\checkpoints\\AlexNet_for_flower.pth'
    assert os.path.exists(
        weights_path), "file '{}' does not exit".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    # infer
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_class = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_index[str(predict_class)],
                                                 predict[predict_class].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_index[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
