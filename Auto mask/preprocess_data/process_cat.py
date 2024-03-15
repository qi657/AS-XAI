import os
import torch
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class Cat(Dataset):

    def __init__(self, root, resize, mode, filename):
        super(Cat, self).__init__()

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.root = root
        self.resize = resize

        self.images, self.labels = self.load_csv(filename)

        if mode == 'train':
            self.images = self.images[:int(0.6 * len(self.images))]
            self.labels = self.labels[:int(0.6 * len(self.labels))]
        elif mode == 'val':
            self.images = self.images[int(0.6 * len(self.images)): int(0.8 * len(self.images))]
            self.labels = self.labels[int(0.6 * len(self.labels)): int(0.8 * len(self.labels))]
        elif mode == 'test':
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]
        else:
            raise Exception('wrong mode')

    def load_csv(self, filename):

        images, labels = [], []

        with open(os.path.join(self.root, filename), 'r') as f:

            records = f.readlines()
            random.shuffle(records)

            for line in records:

                img, label = line.split('\t')
                images.append(os.path.join(self.root, img.strip()))
                labels.append(int(label.strip()))

        assert len(images) == len(labels)

        return images, labels

    def denormalize(self, x_encode):

        mean = torch.tensor(self.mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(self.std).unsqueeze(1).unsqueeze(1)

        img = x_encode * std + mean

        return img

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):


        img, label = self.images[idx], self.labels[idx]

        transform = transforms.Compose([
            lambda x: Image.open(img).convert('RGB'),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        img = transform(img)
        label = torch.tensor(label)

        return img, label


def main():

    import time
    import visdom

    viz = visdom.Visdom()

    data_path = os.path.abspath('D:/datasets/12_kind_cat')
    filename = 'train_list.txt'
    dataset = Cat(data_path, 224, 'train', filename)
    image, label = next(iter(dataset))
    viz.image(dataset.denormalize(image), win='image', opts=dict(title='image'))
    viz.text(str(label.numpy()), win='label', opts=dict(title='label'))

    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    for x, y in loader:

        viz.images(dataset.denormalize(x), nrow=8, win='image', opts=dict(title='image'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='label'))
        time.sleep(10)


if __name__ == '__main__':
    main()
