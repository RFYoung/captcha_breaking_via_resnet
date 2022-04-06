from io import BytesIO
import os
import random
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype

FONT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'font')
DEFAULT_FONTS = (

    os.path.join(FONT_DIR, 'FiraCode-Bold.ttf'),
    os.path.join(FONT_DIR, 'FiraCode-Medium.ttf'),
    os.path.join(FONT_DIR, 'iosevka-ss17-bold.ttf'),
    os.path.join(FONT_DIR, 'iosevka-ss17-medium.ttf'),
    os.path.join(FONT_DIR, 'iosevka-ss17-bolditalic.ttf'),
    os.path.join(FONT_DIR, 'iosevka-ss17-mediumitalic.ttf'),
)

FONT_SIZES = (55, 58, 62, 66, 70)


def random_color(start, end, opacity=None):
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)


class _Captcha(object):
    def generate(self, chars, format='png'):
        """Generate an Image Captcha of the given characters.

        :param chars: text to be generated.
        :param format: image file format
        """
        im = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out

    def write(self, chars, output, format='png'):
        """Generate and write an image CAPTCHA data to the output.

        :param chars: text to be generated.
        :param output: output destination.
        :param format: image file format
        """
        im = self.generate_image(chars)
        return im.save(output, format=format)


class ImageCaptchaEnhanced(_Captcha):
    """Create an image CAPTCHA.

    Many of the codes are borrowed from wheezy.captcha, with a modification
    for memory and developer friendly.

    ImageCaptcha has one built-in font, DroidSansMono, which is licensed under
    Apache License 2. You should always use your own fonts::

        captcha = ImageCaptcha(fonts=['/path/to/A.ttf', '/path/to/B.ttf'])

    You can put as many fonts as you like. But be aware of your memory, all of
    the fonts are loaded into your memory, so keep them a lot, but not too
    many.

    :param width: The width of the CAPTCHA image.
    :param height: The height of the CAPTCHA image.
    :param fonts: Fonts to be used to generate CAPTCHA images.
    :param font_sizes: Random choose a font size from this parameters.
    """
    def __init__(self, width=128, height=64, fonts=None, font_sizes=None):
        self._width = width
        self._height = height
        self._fonts = fonts or DEFAULT_FONTS
        self._font_sizes = font_sizes or FONT_SIZES
        self._truefonts = []
        

    @property
    def truefonts(self):
        if self._truefonts:
            return self._truefonts
        self._truefonts = tuple([
            truetype(n, s)
            for n in self._fonts
            for s in self._font_sizes
        ])
        return self._truefonts

    @staticmethod
    def create_noise_curve(image:Image, color,width=3,number=3):
        w, h = image.size    
        
        for _ in range(0,number):
            x1 = random.randint(-int(w/4), int(w/4))
            x2 = random.randint(int(2*w/4), int(5*w/4))
            if random.randint(0,1) == 1:
                y1 = random.randint(-int(h/3),0)
                y2 = random.randint(int(h/3), int(5*h/6))
                start = random.randint(0, 60)
                end = random.randint(120, 180)
            else:
                y1 = random.randint(int(h/6),int(2*h/3))
                y2 = random.randint(h, int(4*h/3))
                start = random.randint(180, 240)
                end = random.randint(300, 360)
            points = [x1, y1, x2, y2]
            Draw(image).arc(points, start, end, fill=color,width=width)
        return image


    @staticmethod
    def create_noise_dots(image, color, width=2, number=60):
        draw = Draw(image)
        w, h = image.size
        while number:
            x1 = random.randint(0, w)
            y1 = random.randint(0, h)
            x2 = x1 + random.randint(3, 6) * random.choice((1,-1))
            y2 = y1 + random.randint(3, 6)
            draw.line(((x1, y1), (x2, y2)), fill=color, width=width)
            number -= 1
        return image

    def create_captcha_image(self, chars):
        """Create the CAPTCHA image itself.
        The color here is black(1) and white(0) only

        :param chars: text to be generated.
        
        """
        image = Image.new('1', (self._width, self._height),0)
        draw = Draw(image)

        def _draw_character(c):
            font = random.choice(self.truefonts)
            _, _, w, h = draw.textbbox(xy=(0,0), text=c, font=font, anchor='lt')
            im = Image.new('1', (w, h))
            Draw(im).text((0, 0), c, font=font, fill=1,anchor="lt")

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), Image.NEAREST, expand=1)

            return im

        images = []
        for c in chars:
            images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        average = int(text_width / len(chars))
        rand_l = int(0.4 * average)
        rand_r = int(0.2 * average)
        offset = int(average * 0.2)
        x_set = []
        y_set = []
        for im in images:
            w, h = im.size
            x_set.append(offset)
            y = int ((self._height - h)/2)
            y_set.append(y+h)
            image.paste(im, (offset, y),im)
            offset = offset + w + random.randint(-rand_l, rand_r)
        x_set.append(offset)
        if random.randint(0,1) == 0:
            points = (x_set[0]-8, max(y_set)-random.randint(11,15),x_set[-1]+8, min(y_set)-random.randint(16,20))
        else:
            points = (x_set[0]-8, min(y_set)-random.randint(16,20), x_set[-1]+8, max(y_set)-random.randint(11,15))
        Draw(image).line(points,fill=1, width=5)

        if width > self._width:
            image = image.resize((self._width, self._height))

        return image



    def generate_image(self, chars):
        """Generate the image of the given characters.

        :param chars: text to be generated.
        """
        background = random_color(238, 255)
        color = random_color(10,200)
        im = self.create_captcha_image(chars)
        self.create_noise_dots(im, color)
        self.create_noise_curve(im, color)
        im = im.filter(ImageFilter.SMOOTH)
        return im