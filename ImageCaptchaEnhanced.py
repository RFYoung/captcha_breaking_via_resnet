from io import BytesIO
import os
import random
from PIL import Image
from PIL import ImageFilter
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype

import numpy as np
from scipy.ndimage import map_coordinates


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
    """
    Generate random color in RGB

    start : the start range of RGB
    end : the end range of RGB
    """
    red = random.randint(start, end)
    green = random.randint(start, end)
    blue = random.randint(start, end)
    if opacity is None:
        return (red, green, blue)
    return (red, green, blue, opacity)


class _Captcha(object):
    def generate(self, chars, format='png'):
        """
        Generate an Image Captcha of the given characters.

        chars: text to be generated.
        format: image file format
        """
        im = self.generate_image(chars)
        out = BytesIO()
        im.save(out, format=format)
        out.seek(0)
        return out

    def write(self, chars, output, format='png'):
        """
        Generate and write an image CAPTCHA data to the output.

        chars: text to be generated.
        output: output destination.
        format: image file format
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

    width: The width of the CAPTCHA image.
    height: The height of the CAPTCHA image.
    fonts: Fonts to be used to generate CAPTCHA images.
    font_sizes: Random choose a font size from this parameters.
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
        """
        create noise curve on the image
        """
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
        """
        create noise dots on the image
        """
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
            """
            Draw individual characters
            """
            font = random.choice(self.truefonts)
            _, _, w, h = draw.textbbox(xy=(0,0), text=c, font=font, anchor='lt')
            im = Image.new('1', (w, h))
            Draw(im).text((0, 0), c, font=font, fill=1,anchor="lt")

            # rotate
            im = im.crop(im.getbbox())
            im = im.rotate(random.uniform(-30, 30), Image.NEAREST, expand=True)

            return im

        images = []
        for c in chars:
            images.append(_draw_character(c))

        text_width = sum([im.size[0] for im in images])

        width = max(text_width, self._width)
        image = image.resize((width, self._height))

        # draw the line cross the bottom
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


    
    def barrel_distort_waving(self, im:Image, main_color, bg_color, A, B, C, D, X, Y):
        """
        Do barrel distort and waving the captcha
        Using the equation: A^3*r + B^2*r + C*r + D
        X and Y are the center of the distortion
        """
        alpha = random.uniform(0.8,1.2)
        beta = random.randint(-90,90)
        
        wave_shift = lambda line: alpha * (self._width / 30.0) * np.sin( np.pi * (line+beta) / self._height )
        
        m = np.asarray(im).transpose(1,0)
        for row_index in range(self._width):
            m[row_index] = np.roll(m[row_index], int(wave_shift(row_index)))
        m = m.transpose(1,0)
                
        xv,yv = np.meshgrid(np.float32(np.arange(self._width)),np.float32(np.arange(self._height)))

        x_c = self._width/2 if X is None else X
        y_c = self._height/2 if Y is None else Y
        xv = (xv - x_c) / self._width
        yv = (yv - y_c) / self._height
        radius = np.sqrt(xv**2 + yv**2)

        # implement the eqation on "relative radious"
        m_r = A*radius**3 + B*radius**2 + C*radius + D
        
        xv = xv * m_r * self._width + x_c
        yv = yv * m_r * self._height + y_c
        
        xv = xv.astype(np.int16)       
        yv = yv.astype(np.int16)
        
        m = map_coordinates(m, [yv.ravel(),xv.ravel()])
        
        m = np.reshape(m,(self._height,self._width))
        main_color = np.array(main_color)
        bg_color = np.array(bg_color)
        m = np.array([[main_color if i else bg_color for i in m_0] for m_0 in m]).astype(np.uint8)
        im = Image.fromarray(m)
        return  im


    def generate_image(self, chars):
        """
        Generate the image of the given characters.

        chars: text to be generated.
        """
        background = random_color(238, 255)
        color = random_color(10,200)
        im = self.create_captcha_image(chars)
        im = self.barrel_distort_waving(im = im, main_color=color, bg_color=background, A = 0, B = 0.2, C = 0.2, D = 0.9, X = random.randint(int (self._width * 2 / 5), int (self._width * 3 / 5)), Y = random.randint(int (self._height * 2 / 5), int (self._height * 3 / 5)))
        self.create_noise_dots(im, color)
        self.create_noise_curve(im, color)
        im = im.filter(ImageFilter.SMOOTH)
        return im