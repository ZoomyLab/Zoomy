import numpy as np

Rc = 3.66
b = 1.07/2
bed_slope = 0.00083

x0 = np.array([0.,0.,0.])

# outward facing normal w.r.t. the center line
def normal(alpha):
    return np.array([-np.sin(alpha/180*np.pi), -np.cos(alpha/180*np.pi), 0.])
# offset w.r.t. the center line in direction of outward facing normal
def offset(alpha):
    return np.array([b, b, 0.]) * normal(alpha)

def circle_center(alpha):
    return  x0 + normal(alpha) * np.array([Rc, Rc, 0.])

def circle_coords(alpha):
    return np.array([circle_center(alpha) + offset(alpha), circle_center(alpha) - offset(alpha)])

def bed_elevation(alpha, add_length = 0):
    length = alpha/180. * np.pi * Rc + add_length
    return np.array([[0.,0., -bed_slope * length], [0.,0., -bed_slope * length]])


add_1  = 4.1 
add_2  = 4.1 + 1.21 
add_3  = 4.1 + 1.21 + 0.82
add_13 = 1.66
add_14 = 0.87
x_s3 = circle_coords(0) + bed_elevation(0, add_length=add_3)
x_s4 = circle_coords(30)+ bed_elevation(30, add_length=add_3)
x_s5 = circle_coords(60)+ bed_elevation(60, add_length=add_3)
x_s6 = circle_coords(90)+ bed_elevation(90, add_length=add_3)
x_s7 = circle_coords(120)+ bed_elevation(120, add_length=add_3)
x_s8 = circle_coords(150)+ bed_elevation(150, add_length=add_3)
x_s9 = circle_coords(180)+ bed_elevation(180, add_length=add_3)
x_s10 = circle_coords(210)+ bed_elevation(210, add_length=add_3)
x_s11 = circle_coords(240)+ bed_elevation(240, add_length=add_3)
x_s12 = circle_coords(270)+ bed_elevation(270, add_length=add_3)

x_s2 = x_s3 - np.array([[0.82, 0, 0], [0.82, 0, 0]])
x_s2[:,2] = 0
x_s2 += bed_elevation(0, add_length=add_2)
x_s1 = x_s2 - np.array([[1.21, 0, 0], [1.21, 0, 0]])
x_s1[:, 2] = 0
x_s1 += bed_elevation(0, add_length=add_1)
x_in = x_s2 - np.array([[4.1, 0, 0], [4.1, 0, 0]])
x_in[:, 2] = 0
x_s13 = x_s12 - np.array([[0, 1.66, 0], [0, 1.66, 0]]) 
x_s13 += bed_elevation(0., add_length=add_13)
x_out = x_s13 - np.array([[0, 0.87, 0], [0, 0.87, 0]])
x_out += bed_elevation(0., add_length=add_14)


print('in')
print(x_in)
print('1')
print(x_s1)
print('2')
print(x_s2)
print('3')
print(x_s3)
print('4')
print(x_s4)
print('5')
print(x_s5)
print('6')
print(x_s6)
print('7')
print(x_s7)
print('8')
print(x_s8)
print('0')
print(x_s9)
print('10')
print(x_s10)
print('11')
print(x_s11)
print('12')
print(x_s12)
print('13')
print(x_s13)
print('out')
print(x_out)
