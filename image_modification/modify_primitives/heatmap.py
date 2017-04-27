from PIL import Image, ImageDraw


# Draw circle on top of picture
def draw_circle(im, x, y, radius, color):
    draw = ImageDraw.Draw(im)
    draw.ellipse((x-(radius/2), y-(radius/2), x+(radius/2), y+(radius/2)), color)
    del draw
    return im


# Get rgb color from range of values
def rgb(minimum, maximum, value):
    value = -0.5*value + 100
    print(value)
    minimum, maximum = float(minimum), float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


# Save picture
def save_image(im, path):
    im.save(path)


# Show picture
def show_image(im):
    im.show()


#im = Image.open("/home/tommaso/FalsifyNN/pics/roads/desert.jpg")
#col = rgb(0,100,50)
#im = draw_circle(im,1500,1000,10,col)
#im = draw_circle(im,1400,1000,10,col)
#show_image(im)




