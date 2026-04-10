#!/usr/bin/env python3
import glob
import numpy as np
import analysator
import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv) > 1:
    dirname = sys.argv[1]
else:
    dirname = "."

ptinteractive = int(os.environ.get('PTNOINTERACTIVE', '0')) == 0
have_latex    = int(os.environ.get('PTNOLATEX',       '0')) == 0

if ptinteractive: # interactive mode
    try:
        from tqdm import tqdm
        have_tqdm = True
    except ImportError:
        print("WARNING: Could not import tqdm")
        have_tqdm = False
else:
    have_tqdm = False
if not have_tqdm:
    tqdm = lambda x : x

class SI:
    e = 1.6022e-19 #C
    mp = 1.6726e-27 #kg
    me = 9.1094e-31 #kg
    eps0 = 8.8542e-12 # F/m
    mu0 = 4.*np.pi*1e-7 # H/m
    kB = 1.3807e-23 # J/K
    c = 2.9979e8 # m/s

timesteps = []
for filename in glob.glob(dirname+"/bulk*vlsv"):
    parts = filename.split("/")[-1].split(".")
    if len(parts) == 3:
        timesteps.append(int(parts[1]))
timesteps.sort()
tsize = len(timesteps)

for i,t in enumerate(timesteps[:1]):
    f = analysator.vlsvfile.VlsvReader(dirname+"/bulk."+"{:07d}".format(t)+".vlsv")
    [xsize, ysize, zsize] = map(int,f.get_fsgrid_mesh_size()) # uint64t makes some other stuff unhappy

dt = f.read_parameter("dt")
config=f.get_config()

print("dt = "+str(dt)+" s")
dtout = float(config["io"]["system_write_t_interval"][0])
print("dtout = "+str(dtout)+" s")
print("T_end = "+str(timesteps[-1]*dtout)+" s")
xmin = f.read_parameter("xmin")
xmax = f.read_parameter("xmax")
dx = (xmax-xmin)/xsize
print("dx = "+str(dx)+" m")
ymin = f.read_parameter("ymin")
ymax = f.read_parameter("ymax")
dy = (ymax-ymin)/ysize
print("dy = "+str(dy)+" m")

ni_avg = None
print("Processing timesteps")
for i in tqdm(range(len(timesteps))):
    t = timesteps[i]
    if not have_tqdm:
            print("Output step "+str(i)+"/"+str(len(timesteps))+" at time "+str(t))
    f = analysator.vlsvfile.VlsvReader(dirname+"/bulk."+"{:07d}".format(t)+".vlsv")
    cellids = f.read_variable('cellid')
    if f.check_variable('proton/vg_rho'):
        vg_ni = f.read_variable('proton/vg_rho')
        ni = vg_ni[cellids.argsort()].reshape(ysize,xsize)
        ngreen, nred = None, None

        vg_Ui = f.read_variable('proton/vg_v')
        Ui = vg_Ui[cellids.argsort()].reshape(ysize,xsize,3)
    else:
        vg_ngreen = f.read_variable('green/vg_rho')
        vg_nred   = f.read_variable('red/vg_rho')
        ngreen = vg_ngreen[cellids.argsort()].reshape(ysize,xsize)
        nred   = vg_nred[cellids.argsort()].reshape(ysize,xsize)
        ni = ngreen + nred

        vg_Ugreen = f.read_variable('green/vg_v')
        vg_Ured   = f.read_variable('red/vg_v')
        Ugreen = vg_Ugreen[cellids.argsort()].reshape(ysize,xsize,3)
        Ured   = vg_Ured[cellids.argsort()].reshape(ysize,xsize,3)
        Ui = np.zeros( (ni.shape[0],ni.shape[1],3) )
        Ui[:,:,0] = (ngreen*Ugreen[:,:,0]+nred*Ured[:,:,0])/(ngreen+nred)
        Ui[:,:,1] = (ngreen*Ugreen[:,:,1]+nred*Ured[:,:,1])/(ngreen+nred)
        Ui[:,:,2] = (ngreen*Ugreen[:,:,2]+nred*Ured[:,:,2])/(ngreen+nred)

    if ni_avg is None:
        ni_avg = np.average(ni)

    if have_latex:
        title = "$t = "+str(round(t*dtout,2))+"$s"
    else:
        title = "t = "+str(round(t*dtout,2))+"s"

    plt.title(title)
    im = plt.imshow(ni, extent=(xmin,xmax,ymin,ymax), origin="lower", aspect="auto")
    if have_latex:
        plt.colorbar(im, label=r"$n_i \,/\, m^{-3}$")
    else:
        plt.colorbar(im, label="n_i / m^-3")
    plt.contour(ni, extent=(xmin,xmax,ymin,ymax), levels=[ni_avg], colors="black", linewidths=0.5)
    plt.xlabel("x / m")
    plt.ylabel("y / m")
    plt.savefig(dirname+"/ni_"+str(t)+".png")
    plt.close()

    if ngreen is not None:
        plt.title(title)
        im = plt.imshow(ngreen, extent=(xmin,xmax,ymin,ymax), origin="lower", aspect="auto", cmap="Greens")
        if have_latex:
            plt.colorbar(im, label=r"$n_i \,/\, m^{-3}$")
        else:
            plt.colorbar(im, label="n_i / m^-3")
        plt.contour(ni, extent=(xmin,xmax,ymin,ymax), levels=[ni_avg], colors="black", linewidths=0.5)
        plt.xlabel("x / m")
        plt.ylabel("y / m")
        plt.savefig(dirname+"/ngreen_"+str(t)+".png")
        plt.close()

    if nred is not None:
        plt.title(title)
        im = plt.imshow(nred, extent=(xmin,xmax,ymin,ymax), origin="lower", aspect="auto", cmap="Reds")
        if have_latex:
            plt.colorbar(im, label=r"$n_i \,/\, m^{-3}$")
        else:
            plt.colorbar(im, label="n_i / m^-3")
        plt.contour(ni, extent=(xmin,xmax,ymin,ymax), levels=[ni_avg], colors="black", linewidths=0.5)
        plt.xlabel("x / m")
        plt.ylabel("y / m")
        plt.savefig(dirname+"/nred"+str(t)+".png")
        plt.close()

    if nred is not None:
        plt.title(title)
        im = plt.imshow(nred/ni, extent=(xmin,xmax,ymin,ymax), origin="lower", aspect="auto", cmap="RdYlGn_r")
        if have_latex:
            plt.colorbar(im, label=r"$n_{red} \,/\, n_i$")
        else:
            plt.colorbar(im, label="n_red / n_i")
        plt.contour(nred/ni, extent=(xmin,xmax,ymin,ymax), levels=[0.2,0.5,0.8], colors="black", linewidths=0.5)
        plt.xlabel("x / m")
        plt.ylabel("y / m")
        plt.savefig(dirname+"/mixing"+str(t)+".png")
        plt.close()

    # plt.title(title)
    # im = plt.imshow(Ui[:,:,0], extent=(xmin,xmax,ymin,ymax), origin="lower", aspect="auto")
    # if have_latex:
    #     plt.colorbar(im, label=r"$U_x \,/\, m/s$")
    # else:
    #     plt.colorbar(im, label="U_x / m/s")
    # plt.contour(ni, extent=(xmin,xmax,ymin,ymax), levels=[ni_avg], colors="black", linewidths=0.5)
    # plt.xlabel("x / m")
    # plt.ylabel("y / m")
    # plt.savefig(dirname+"/Ux_"+str(t)+".png")
    # plt.close()

    # plt.title(title)
    # im = plt.imshow(Ui[:,:,1], extent=(xmin,xmax,ymin,ymax), origin="lower", aspect="auto")
    # if have_latex:
    #     plt.colorbar(im, label=r"$U_y \,/\, m/s$")
    # else:
    #     plt.colorbar(im, label="U_y / m/s")
    # plt.contour(ni, extent=(xmin,xmax,ymin,ymax), levels=[ni_avg], colors="black", linewidths=0.5)
    # plt.xlabel("x / m")
    # plt.ylabel("y / m")
    # plt.savefig(dirname+"/Uy_"+str(t)+".png")
    # plt.close()
