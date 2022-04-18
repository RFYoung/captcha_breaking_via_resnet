from ImageCaptchaEnhanced import ImageCaptchaEnhanced

# test the captcha with some potential confusing characters
captcha = ImageCaptchaEnhanced(width=192,height=64)
captcha.write("QO1D", "test/demo1.png")
captcha.write("OQ1D", "test/demo2.png")
captcha.write("0D17", "test/demo3.png")
captcha.write("0Dq9", "test/demo4.png")
captcha.write("Q96g", "test/demo5.png")



