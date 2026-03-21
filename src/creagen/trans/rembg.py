from PIL import Image
from rembg import remove

input = Image.open("input.png")
output = remove(input)
output.save("output.png")
