import matplotlib.pyplot as plt
import numpy as np
import os
import tol_colors as tc
import seaborn as sns
import pyswashes

from library.python.misc.io import load_fields_from_hdf5, load_mesh_from_hdf5


sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
main_dir = os.getenv("ZOOMY_DIR")

sns.set_palette(tc.colorsets['bright'])
def vam_analytical_eta(): 
    x = [-0.5928667563930013, -0.5430686406460297, -0.4946164199192463, -0.4448183041722746, -0.3990578734858681, -0.34522207267833105, -0.2981157469717362, -0.2510094212651413, -0.19851951547779273, -0.15141318977119783, -0.10430686406460293, -0.0531628532974428, -0.0006729475100942239, 0.04643337819650073, 0.09757738896366086, 0.14737550471063254, 0.19851951547779279, 0.24562584118438757, 0.29811574697173626, 0.34791386271870794, 0.3963660834454913, 0.446164199192463, 0.49865410497981155, 0.5511440107671601]
    y = [0.3418918918918919, 0.34121621621621623, 0.3398648648648649, 0.3418918918918919, 0.3398648648648649, 0.3398648648648649, 0.33851351351351355, 0.33783783783783783, 0.3337837837837838, 0.32770270270270274, 0.322972972972973, 0.31486486486486487, 0.3054054054054054, 0.29054054054054057, 0.26891891891891895, 0.2425675675675676, 0.21621621621621623, 0.18581081081081083, 0.15540540540540543, 0.13108108108108107, 0.10608108108108108, 0.0918918918918919, 0.07297297297297298, 0.06554054054054054]
    return x, y

def vam_analytical_p():
    x = [-0.6001390820584145, -0.5556328233657858, -0.5041724617524339, -0.4485396383866481, -0.3998609179415855, -0.35396383866481224, -0.30389429763560505, -0.24965229485396384, -0.2051460361613352, -0.15229485396383868, -0.1008344923504868, -0.05354659248956889, -0.0006954102920723737, 0.0521557719054242, 0.09805285118219742, 0.15090403337969394, 0.20236439499304582, 0.24826147426981915, 0.30111265646731566, 0.3539638386648122, 0.4026425591098748, 0.45271210013908203, 0.5013908205841446, 0.5514603616133518, -0.15229485396383868]
    y = [0.3319327731092437, 0.33053221288515405, 0.32212885154061627, 0.30952380952380953, 0.29061624649859946, 0.27450980392156865, 0.25, 0.22338935574229693, 0.18417366946778713, 0.15756302521008403, 0.12394957983193278, 0.09453781512605042, 0.0700280112044818, 0.04201680672268908, 0.04481792717086835, 0.03431372549019608, 0.04831932773109244, 0.058823529411764705, 0.06022408963585434, 0.06932773109243698, 0.0742296918767507, 0.0861344537815126, 0.08473389355742297, 0.07913165266106442, 0.15756302521008403]
    return x, y

def plot_vam2():
    fig_width = 372 / 72
    fig_height = fig_width * 0.9  # slightly taller for legends

    path = os.path.join(main_dir, 'outputs/vam/VAM.h5')
    Q, Qaux, time = load_fields_from_hdf5(path)
    mesh = load_mesh_from_hdf5(path)

    n_inner = mesh.n_inner_cells
    x = mesh.cell_centers[0, :n_inner]
    h = Q[0, :n_inner]
    b = Q[5, :n_inner]
    p1 = Qaux[2, :n_inner]
    g = 9.81
    pb = (h * g + 2*p1)/g

    # Use subplot_mosaic to arrange plots and their legends below
    mosaic = """
    AB
    ab
    """
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(fig_width, fig_height), constrained_layout=True, gridspec_kw={"height_ratios": [3, 1]} )

    ax_eta = axes["A"]
    ax_p = axes["B"]
    leg_eta = axes["a"]
    leg_p = axes["b"]

    # Analytical data
    a_x_eta, a_eta = vam_analytical_eta()
    a_x_p, a_p = vam_analytical_p()

    # Plot eta (left)
    l1 = ax_eta.plot(x, h + b, label=r'$\eta=h+b$')[0]
    l2 = ax_eta.plot(x, b, label=r'$b$')[0]
    l3 = ax_eta.plot(a_x_eta, a_eta, 'k*', label=r'$\eta^{exp}$')[0]
    ax_eta.set_xlabel(r'$x \text{ in } (m)$')
    ax_eta.set_ylabel(r'$\eta \; | \; b  \; \text{ in } (m)$')
    ax_eta.set_ylim(0, 0.4)

    # Plot p_b (right)
    l4 = ax_p.plot(x, pb, label=r'$p_{b}/(\rho \; g)$')[0]
    l5 = ax_p.plot(a_x_p, a_p, 'k*', label=r'$p_{b}/(\rho \;g)^{exp}$')[0]
    ax_p.set_xlabel(r'$x \text{ in } (m)$')
    ax_p.set_ylabel(r'$p_{b}/(\rho \;g) \text{ in } (m)$')
    ax_p.set_ylim(0, 0.4)

    # Hide legend axes borders
    leg_eta.axis("off")
    leg_p.axis("off")

    # Add fitted legends to each box
    leg_eta.legend([l1, l2, l3],
                   [l.get_label() for l in [l1, l2, l3]],
                   loc='center',
                   frameon=True)

    leg_p.legend([l4, l5],
                 [l.get_label() for l in [l4, l5]],
                 loc='center',
                 frameon=True)

    fig.suptitle('VAM: flow over bump')

    return fig


def plot_vam(path='outputs/vam/VAM.h5'):
    #fig_width =  372 / 72
    fig_width =  472 / 72
    fig_height = fig_width * 0.618
    path = os.path.join(main_dir, path)
    Q, Qaux, time = load_fields_from_hdf5(path)
    mesh = load_mesh_from_hdf5(path)
    n_inner = mesh.n_inner_cells
    x = mesh.cell_centers[0, :n_inner]
    h = Q[0, :n_inner]
    b = Q[5, :n_inner]
    p1 = Qaux[2, :n_inner]
    g = 9.81
    pb = (h * g + 2*p1)/g
    
    fig, ax = plt.subplots(1,2, figsize=(fig_width, fig_height), constrained_layout=True, sharey=True)
    a_x_eta, a_eta = vam_analytical_eta()
    a_x_p, a_p = vam_analytical_p()
    #ax[0].plot(x, h+b, label=r'$\eta=h+b$')
    ax[0].plot(x, h+b, label=r'$\eta$')
    ax[0].plot(x, b, label=r'$b$')
    ax[0].plot(a_x_eta, a_eta, 'k*', label=r'$\eta^{exp}$')
    ax[0].set_xlabel(r'$x \text{ in } (m)$')
    #ax[0].set_ylabel(r'$\eta \; | \; b  \; \text{ in } (m)$')
    #ax[0].set_ylabel(r'$\eta \; | \; b \; | p^*  \; \text{ in } (m)$')
    ax[0].set_ylabel(r'$\eta \; | \; b  \; | \; p_{b}/(\rho \;g) \; \text{ in } (m)$')

    ax[1].plot(x, pb, label=r'$p_{b}/(\rho \; g)$')
    ax[1].plot(a_x_p, a_p, 'k*', label=r'$p_{b}/(\rho \;g)^{exp}$')
    #ax[1].plot(x, pb, label=r'$p^*$')
    #ax[1].plot(a_x_p, a_p, 'k*', label=r'$p^*$')
    ax[1].set_xlabel(r'$x \text{ in } (m)$')
    # ax[1].set_ylabel(r'$p_{b}/(\rho \;g) \text{ in } (m)$')

    ax[0].set_ylim(0, 0.4)
    ax[1].set_ylim(0, 0.4)
    
    fig.suptitle('VAM: flow over bump')
    ax[0].legend()
    ax[1].legend()
    return fig


def analytical_swe():
    s = pyswashes.OneDimensional(3, 1, 1, 50)
    
    print(s.dom_params)
    
    x = np.linspace(0, s.dom_params['length'], int(s.dom_params['ncellx']))
    b = s.np_array('gd_elev')
    h = s.np_array('depth')
    u = s.np_array('u')
    #v = s.np_array('v')
    return x, b, h, u

def plot_swe(path='outputs/sme_0/ShallowMoments.h5'):
    #fig_width =  372 / 72
    fig_width =  472 / 72
    fig_height = fig_width * 0.618
    path = os.path.join(main_dir, path )
    Q, Qaux, time = load_fields_from_hdf5(path, i_snapshot=-1)
    mesh = load_mesh_from_hdf5(path)
    n_inner = mesh.n_inner_cells
    x = mesh.cell_centers[0, :n_inner]
    h = Q[0, :n_inner]
    u = Q[1, :n_inner] / h
    g = 9.81
    
    fig, ax = plt.subplots(1,2, figsize=(fig_width, fig_height), constrained_layout=True, sharey=False)
    a_x, a_b, a_h, a_u = analytical_swe()
    ax[0].plot(x, h, label=r'$h$')
    ax[0].plot(a_x, a_h, 'k*', label=r'$h^{ana}$')
    ax[0].set_xlabel(r'$x \text{ in } (m)$')
    ax[0].set_ylabel(r'$h \text{ in } (m)$')

    ax[1].plot(x, u, label=r'$u$')
    ax[1].plot(a_x, a_u, 'k*', label=r'$u^{ana}$')
    ax[1].set_xlabel(r'$x \text{ in } (m)$')
    ax[1].set_ylabel(r'$u \text{ in } (m/s)$')

    fig.suptitle('SWE: channel without bottom and without friction')
    ax[0].legend()
    ax[1].legend()
    return fig

def plot_poisson(path='outputs/poisson/Poisson.h5'):

    def sol(x):
        return x**2 +1
    fig_width =  472 / 72
    fig_height = fig_width * 0.618
    path = os.path.join(main_dir, path)
    Q, Qaux, time = load_fields_from_hdf5(path)
    mesh = load_mesh_from_hdf5(path)
    n_inner = mesh.n_inner_cells
    x = mesh.cell_centers[0, :n_inner]
    T = Q[0, :n_inner]

    x_sol = np.linspace(0, 1, 30)
    T_sol = sol(x_sol)
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True, sharey=False)
    ax.plot(x, T, label=r'$T(x)$')
    ax.plot(x_sol, T_sol, 'k*', label=r'$T(x)^{ana}$')
    ax.set_xlabel(r'$x \text{ in } (m)$')
    ax.set_ylabel(r'$T \text{ in } (K)$')

    fig.suptitle('Poisson equation in one dimension ')
    ax.legend()
    return fig

if __name__=='__main__':
    #fig = plot_vam()
    #fig.savefig("vam.pdf", dpi=600)
    #plt.show()

    #fig = plot_swe()
    #fig.savefig("swe.pdf", dpi=600)
    #plt.show()

    fig = plot_poisson()
    fig.savefig("poisson.pdf", dpi=600)
    #plt.show()

