#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

# constants
mp = 1.672622e-27 # kg
me = 9.109400e-31 # kg
kB = 1.380649e-23 # J/K
q =  1.602177e-19 # C

# velocity-space block width (WID) is not in the cfg file
block_width = 4

# mass_units values that can appear in a [<pop>_properties] section
MASS_UNITS = {"ELECTRON": me, "PROTON": mp}


def parse_cfg(path):
    """
    Parse a Vlasiator-style .cfg file into a dict of dicts
    {section_name: {key: [values]}}.

    Keys that appear before the first '[section]' header are stored under the
    section name "". Values are kept as lists because the format allows a key
    to be repeated such as ParticlePopulations.'#' starts a comment till the
    end of the line
    """
    sections = {"": {}}
    current = sections[""]

    with open(path, "r") as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.split("#")[0].strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current = sections.setdefault(line[1:-1].strip(), {})
                continue
            if "=" not in line:
                raise ValueError(f"{path}:{lineno}: cannot parse line: {raw_line!r}")
            key, value = (part.strip() for part in line.split("=", 1))
            current.setdefault(key, []).append(value)

    return sections

def populations_from_cfg(path):
    """
    Read a Vlasiator-style .cfg file and yield (pop_name, params) for every
    population it lists, where params is a dict of keyword arguments ready
    to pass to plot_vdf().

    density/temperature/VX0 are read from the population's project-specific
    section, [<pop>_<project>] (e.g. [proton_Dispersion])
    'project' is read from the top of the file (e.g. 'project = Dispersion').
    """
    cfg = parse_cfg(path)
    top = cfg[""]

    if "ParticlePopulations" not in top:
        raise ValueError(f"{path}: no 'ParticlePopulations' entries found")
    project = top["project"][0]

    for pop in top["ParticlePopulations"]:
        needed = {
            "properties": f"{pop}_properties",
            "vspace": f"{pop}_vspace",
            "sparse": f"{pop}_sparse",
            "init": f"{pop}_{project}",
        }
        secs = {}
        for role, name in needed.items():
            if name not in cfg:
                raise ValueError(f"{path}: population '{pop}' is missing section [{name}]")
            secs[role] = cfg[name]

        mass_value = float(secs["properties"]["mass"][0])
        mass_units = secs["properties"]["mass_units"][0]
        if mass_units not in MASS_UNITS:
            raise ValueError(f"{path}: population '{pop}' has unrecognized mass_units '{mass_units}' (expected one of {MASS_UNITS})")

        params = dict(
            particle_mass      = mass_value * MASS_UNITS[mass_units],
            density            = float(secs["init"]["rho"][0]),
            temperature        = float(secs["init"]["Temperature"][0]),
            vmean              = float(secs["init"]["VX0"][0]),
            vmin               = float(secs["vspace"]["vx_min"][0]),
            vmax               = float(secs["vspace"]["vx_max"][0]),
            n_blocks           = int(secs["vspace"]["vx_length"][0]),
            sparsity_threshold = float(secs["sparse"]["minValue"][0]),
        )

        yield pop, params


def plot_vdf(particle_mass, density, temperature, vmean, vmin, vmax,
             n_blocks, sparsity_threshold, pop_name=None):
    """Draw one VDF-parameters figure (new window) for a single population."""

    v_th = math.sqrt(3. * kB * temperature / particle_mass)

    fig = plt.figure(pop_name)
    ax = fig.gca()

    dv = (vmax-vmin)/(n_blocks * block_width)
    v = np.arange(vmin, vmax, dv)

    vdf = np.ma.array(density * (particle_mass / (2.*math.pi*kB*temperature))**(3./2.) * np.exp(-particle_mass*(v-vmean)**2 / (2.*kB*temperature)))
    vdf = np.ma.masked_where(vdf < 0.005*sparsity_threshold, vdf)

    print(f"Parameters for population '{pop_name}':" if pop_name is not None else "Parameters:")
    print(f"mass: {particle_mass:.3e} kg or {particle_mass/mp:.1f} proton masses")
    print(f"temperature: {temperature:.3e} K or {temperature * kB / q:.3e} eV")
    print(f"density: {density:.3e} m^-3")
    print(f"thermal speed: {v_th:.3e} m/s")
    print(f"dv: {dv:.3e} m/s")
    print(f"VDF max, sparsity threshold {np.max(vdf):.2e}, {sparsity_threshold:.1e}")


    ax.scatter(v, vdf, s=10, label=f"v-space cell resolution dv = {dv:.3e} m/s")
    #ax.step(v, vdf)
    ax.scatter(v[::block_width], vdf[::block_width], marker="|", s=300, label="v-space blocks WID = "+str(block_width)+" cells")
    #ax.step(v[::block_width], vdf[::block_width], where="post")
    ax.axvline(vmean, label=f"mean velocity {vmean:.1e} m/s")
    ax.set_yscale("log")
    ax.hlines(sparsity_threshold, vmin, vmax, label=f"sparsity threshold {sparsity_threshold:.1e} s^3/m^6", color="C3", lw=3, alpha=0.5)
    ax.axvspan(vmean - v_th, vmean + v_th, alpha=0.3, label=f"thermal velocity = {v_th:.3e} m/s")
    ax.axhspan(np.ma.min(vdf)*0.1, sparsity_threshold, xmin=vmin, xmax=vmax, color="C3", hatch="/", alpha=0.5)
    ax.set_xlim((vmin, vmax))

    ax.set_xlabel("Velocity space extent (m/s)")
    ax.set_ylabel("Phase-space density of VDF (s^3/m^6)")
    title = f"mass {particle_mass:.3e} kg or {particle_mass/me:.1f} electron masses\n n = {density:.3e} m^-3 and T = {temperature:.3e} K or {temperature * kB / q:.3e} eV"
    ax.set_title(f"'{pop_name}': {title}" if pop_name is not None else title)

    ax.legend()

    plt.draw()


def main():
    if len(sys.argv) > 2:
        sys.exit(f"usage: {sys.argv[0]} [config.cfg]")

    if len(sys.argv) == 2:
        cfg_path = sys.argv[1]
        for pop_name, params in populations_from_cfg(cfg_path):
            plot_vdf(pop_name=pop_name, **params)
    else:
        # enter parameters in SI!!!
        plot_vdf(
            particle_mass = 1.*mp,      # kg
            density = 1e6,              # m^-3
            temperature = 5e5,          # K
            vmean = 5e5,                # m/s
            vmin = -1.2e6,              # m/s
            vmax =  1.2e6,              # m/s
            n_blocks = 15,
            sparsity_threshold = 1e-15, # s^3/m^6
        )

    plt.show()


if __name__ == "__main__":
    main()
