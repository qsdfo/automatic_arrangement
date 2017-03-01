import math
import numpy as np

from bokeh.plotting import figure, show, output_file

N = 500
x = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
xx, yy = np.meshgrid(x, y)


# Here is z = f(x,y)
def gaussienne(xx, yy, sigx, sigy, mux, muy):
    # 2dim gaussian with diagonal covariance matrix
    xx_term = -(xx-mux)**2 / (2*sigx**2)
    yy_term = -(yy-muy)**2 / (2*sigy**2)
    z = np.exp(xx_term + yy_term) / ((2 * math.pi) * math.sqrt(sigx * sigy))
    return z

sigma = 0.1
mux1 = 0.3
muy1 = 0.1
mux2 = 0.6
muy2 = 0.7

d = (5./8) * gaussienne(xx,yy,sigma,sigma,mux1,muy1) +\
    (3./8) * gaussienne(xx,yy,sigma,sigma,mux2,muy2)


p = figure(x_range=(0, 1), y_range=(0, 1))

# must give a vector of image data for image parameter
p.image(image=[d], x=0, y=0, dw=1, dh=1, palette="Spectral11")

output_file("image.html", title="image.py example")

show(p)  # open a browser
