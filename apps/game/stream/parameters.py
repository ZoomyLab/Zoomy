

Nx = 600
Ny = 600
n_ghosts = 5

n_timesteps = 10

o_in = [[300, 400]]

n_gauges_out = 2
o_out = [[200, 300], [400, 500]]
n_gauges_top = 2
o_top = [[200, 300], [400, 450]]
n_gauges_bot = 1
o_bot = [[300, 400]]

def convert_to_wall(l):
    out = []
    o_old = 0 
    if len(l) > 0:
        for i in range(len(l)):
            [o0, o1] = l[i]
            out.append([o_old, o0])
            o_old = o1
    out.append([o_old, Nx])
    return out

            



