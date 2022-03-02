import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import string

from CaptchaDataset import CaptchaDataset
from Cnn import CnnModel
from ResNet import ResNetModel as ResNet50Model
from ResNet2 import ResNetModel as ResNet34Model

from train_valid_test import train, valid, test


CHAR_SET = [string.digits + string.ascii_uppercase, string.digits + string.ascii_lowercase]

CHAR_SET = list(map(list, zip(*CHAR_SET)))



PIC_WIDTH, PIC_HEIGHT, CAPTCHA_STR_LEN, CAPTCHA_CHAR_LEN = 192, 64, 4, len(CHAR_SET)
BATCH_SIZE = 128

# model = CnnModel(CAPTCHA_CHAR_LEN, CAPTCHA_STR_LEN, input_shape=(3, PIC_HEIGHT, PIC_WIDTH))
# model = ResNet50Model(CAPTCHA_CHAR_LEN, CAPTCHA_STR_LEN, input_shape=(3, PIC_HEIGHT, PIC_WIDTH))
model = ResNet34Model(CAPTCHA_CHAR_LEN, CAPTCHA_STR_LEN, input_shape=(3, PIC_HEIGHT, PIC_WIDTH))

STORE_DIRECTORY = "log/resnet2_2/"

os.makedirs(STORE_DIRECTORY+"pic/", exist_ok=False)

model = model.cuda()

train_set = CaptchaDataset(CHAR_SET, 1000 * BATCH_SIZE, PIC_WIDTH, PIC_HEIGHT, CAPTCHA_STR_LEN)
valid_set = CaptchaDataset(CHAR_SET, 100 * BATCH_SIZE, PIC_WIDTH, PIC_HEIGHT, CAPTCHA_STR_LEN)
test_set = CaptchaDataset(CHAR_SET, 100 * BATCH_SIZE, PIC_WIDTH, PIC_HEIGHT, CAPTCHA_STR_LEN)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=16)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, num_workers=16)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, num_workers=16)



writer = SummaryWriter(STORE_DIRECTORY+"tensorboard")
log_file = open(STORE_DIRECTORY+"logs.txt", 'a+')

optimizer = torch.optim.Adam(model.parameters(), 1e-4, amsgrad=True)
epochs = 25
for epoch in range(0, epochs):
    train(model, optimizer, epoch, train_loader, writer)
    valid(model, epoch, valid_loader, writer)

optimizer = torch.optim.Adam(model.parameters(), 1e-5, amsgrad=True)
small_lr_epochs = 25
for epoch in range(epochs, epochs+small_lr_epochs):
    train(model, optimizer, epoch, train_loader, writer)
    valid(model, epoch, valid_loader, writer)

test_epochs = 10
test(model, test_epochs, test_loader, writer, log_file)

    
def decode(sequence):
    return ''.join([CHAR_SET[x][0] for x in sequence])


model.eval()
final_dataset = CaptchaDataset(CHAR_SET, 1, PIC_WIDTH, PIC_HEIGHT, CAPTCHA_STR_LEN)
for _ in range(10):
    with torch.no_grad():
        do = True
        while do:
            image, target = final_dataset[0]
            target = target.cpu().numpy()
            
            print('true:', decode(target), file=log_file)
            output = model(image.unsqueeze(0).cuda())
            output_argmax = output.detach().argmax(dim=-1).cpu().numpy()
            print('pred:', decode(output_argmax[0]), file=log_file)
            do = (decode(target) == decode(output_argmax[0]))


    im = torchvision.transforms.ToPILImage()(image)
    im.save(STORE_DIRECTORY + "pic/" +decode(target) + "_" + decode(output_argmax[0]) + ".png")
    

inputs = torch.zeros((1,3, PIC_HEIGHT, PIC_WIDTH)).cuda()
out = model(inputs)



log_file.close()
writer.close()