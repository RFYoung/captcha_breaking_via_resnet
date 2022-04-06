from xml.dom.minidom import CharacterData
import torch, random
from torchvision.transforms.functional import to_tensor
from torch.utils.data.dataset import Dataset
from ImageCaptchaEnhanced import ImageCaptchaEnhanced


class CaptchaDataset(Dataset):
    def __init__(self, characters, length, width, height, label_length):
        super(CaptchaDataset, self).__init__()
        
        self.characters = characters
        self.length = length
        self.width = width
        self.height = height
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptchaEnhanced(width=width, height=height)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # print(self.characters)
        random_str_index_set = [random.randrange(0,self.n_class) for _ in range(0,self.label_length)]
        img = "".join([self.characters[x][random.randint(0,1)] for x in random_str_index_set])
        image = to_tensor(self.generator.generate_image("".join(img)))
        target = torch.tensor(random_str_index_set, dtype=torch.int64)
        return image, target
