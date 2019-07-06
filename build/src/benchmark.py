from edo import finite_difference
import numpy
import PIL

data_dir = '.'
which = 'camera.png'

image = numpy.array(Image.open(os.path.join(data_dir, which)).convert('L'))