
scale = 3
Nx = 60*scale
Ny = 60*scale
n_ghosts = 5

q_in = 0.01
h_in = 0.1

n_timesteps = 20
end_time = 60

o_in = [[int(scale*35), int(scale*45)]]

n_gauges_out = 2
o_out = [[int(scale*20), int(scale*30)], [int(scale*45), int(scale*55)]]
n_gauges_top = 2
o_top = [[int(scale*15), int(scale*25)], [int(scale*40), int(scale*45)]]
n_gauges_bot = 1
o_bot = [[int(scale*30), int(scale*40)]]

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

            



