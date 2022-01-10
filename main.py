import torch
from torch.utils.data import DataLoader
import string

from CaptchaDataset import CaptchaDataset
from Cnn import CnnModel
from ResNet import ResNetModel
from train_valid import train, valid


CHAR_SET = [string.digits + string.ascii_uppercase, string.digits + string.ascii_lowercase]

CHAR_SET = list(map(list, zip(*CHAR_SET)))



PIC_WIDTH, PIC_HEIGHT, CAPTCHA_STR_LEN, CAPTCHA_CHAR_LEN = 192, 64, 4, len(CHAR_SET)
BATCH_SIZE = 128

# model = CnnModel(CAPTCHA_CHAR_LEN, CAPTCHA_STR_LEN, input_shape=(3, PIC_HEIGHT, PIC_WIDTH))
model = ResNetModel(CAPTCHA_CHAR_LEN, CAPTCHA_STR_LEN, input_shape=(3, PIC_HEIGHT, PIC_WIDTH))


model = model.cuda()

train_set = CaptchaDataset(CHAR_SET, 1000 * BATCH_SIZE, PIC_WIDTH, PIC_HEIGHT, CAPTCHA_STR_LEN)
valid_set = CaptchaDataset(CHAR_SET, 100 * BATCH_SIZE, PIC_WIDTH, PIC_HEIGHT, CAPTCHA_STR_LEN)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=16)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, num_workers=16)



optimizer = torch.optim.Adam(model.parameters(), 1e-4, amsgrad=True)
epochs = 25
for epoch in range(0, epochs):
    train(model, optimizer, epoch, train_loader)
    valid(model, epoch, valid_loader)
