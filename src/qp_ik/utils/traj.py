import numpy as np


def gen_line_traj_constant_v(start_ratio, length, vel, dt, steps):
    start_ratio = start_ratio % 1
    half_period = int(length/vel/dt)
    l_up = np.arange(half_period)*dt*vel
    l_down = np.flip(l_up)
    l_full = np.concatenate([l_down,l_up])
    n_periods = steps//len(l_full)
    n_rest = steps%len(l_full)
    l_full = np.roll(l_full, shift=-int(len(l_full)*start_ratio))
    l_full -= length*0.5
    l_final = np.concatenate([l_full[None,:].repeat(n_periods, axis=0).flatten(), l_full[:n_rest]])
    return l_final

def gen_cos_line_traj(start_ratio, max_length, max_vel, dt, steps):
    t = np.arange(0, steps)
    x0 = start_ratio * np.pi
    x = t*dt*max_vel
    return np.cos(x*max_vel/max_length+x0)*max_length