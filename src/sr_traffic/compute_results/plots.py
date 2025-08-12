import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from dctkit.dec import cochain as C
import os
import jax.numpy as jnp
import matplotlib.colors as mcolors


def stgp_traffic_plots(
    gpsr,
    S,
    flats,
    test_data,
    density,
    v,
    f,
    t_sampled_circ,
    step,
    output_path,
):
    os.chdir(output_path)

    x_sampled = S.node_coords
    x_sampled_circ = (x_sampled[1:] + x_sampled[:-1]) / 2

    def flat_left_wrap(x):
        return flats["linear_left"](C.CochainD0(S, x)).coeffs

    flat_left = vmap(flat_left_wrap)

    sols = gpsr.predict(test_data)

    rho_comp = sols["rho"][:, ::step]
    v_comp = sols["v"][:, ::step]
    f_comp = sols["f"][:, ::step]

    # initial and boundary values of v_comp and
    # f_comp are known (ic+ bc on rho)
    v_comp = v_comp.at[:, 0].set(v[:, 0])
    v_comp = v_comp.at[0, :].set(v[0, :])
    v_comp = v_comp.at[-3:, :].set(v[-3:, :])

    f_comp = f_comp.at[:, 0].set(f[:, 0])
    f_comp = f_comp.at[0, :].set(f[0, :])
    f_comp = f_comp.at[-3:, :].set(f[-3:, :])

    flat_rho = C.CochainD1(S, flat_left(rho_comp.T)[:, :, 0].T)
    rho_computed = C.star(flat_rho).coeffs[:-1]

    flat_density = C.CochainD1(S, flat_left(density.T)[:, :, 0].T)
    density_data = C.star(flat_density).coeffs[:-1]

    # errors
    rho_error = jnp.sqrt(jnp.sum(jnp.square(density - rho_comp))) / jnp.sqrt(
        jnp.sum(jnp.square(density))
    )
    v_error = jnp.sqrt(jnp.sum(jnp.square(v - v_comp))) / jnp.sqrt(
        jnp.sum(jnp.square(v))
    )
    f_error = jnp.sqrt(jnp.sum(jnp.square(f - f_comp))) / jnp.sqrt(
        jnp.sum(jnp.square(f))
    )

    # tts
    tts_data = np.trapz(
        np.trapz(density, t_sampled_circ, axis=1),
        x_sampled_circ.flatten(),
        axis=0,
    )
    tts = np.trapz(
        np.trapz(rho_comp, t_sampled_circ, axis=1),
        x_sampled_circ.flatten(),
        axis=0,
    )

    error_tts = np.abs((tts - tts_data) / tts_data)

    print(rho_error, v_error, f_error, error_tts)

    plt.scatter(
        density_data[1:-3, 1:].flatten(),
        f[1:-3, 1:].flatten(),
        marker=".",
        c="#4757fb",
        label="Data",
    )
    plt.scatter(
        rho_computed[1:-3, 1:].flatten(),
        f_comp[1:-3, 1:].flatten(),
        marker=".",
        c="#ff0000",
        label="SR-Traffic",
    )
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$\rho V(\rho)$")
    plt.legend()
    plt.savefig("flux.png", dpi=300)

    plt.clf()

    plt.figure(1, figsize=(20, 10))

    _, axes = plt.subplots(3, 2, num=1)

    x_mesh, t_mesh = np.meshgrid(x_sampled_circ[1:-3], t_sampled_circ)

    vmin = np.min(density.T)
    vmax = np.max(density.T)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = "rainbow"

    # plot rho
    rho_plot = axes[0, 0].contourf(
        t_mesh, x_mesh, density[1:-3].T, levels=100, cmap=cmap, norm=norm
    )
    axes[0, 0].set_title("Data")
    rho_computed_plot = axes[0, 1].contourf(
        t_mesh, x_mesh, rho_comp[1:-3].T, levels=100, cmap=cmap, norm=norm
    )
    axes[0, 1].set_title("SR-Traffic")
    plt.colorbar(
        rho_plot, label=r"$\rho$", ax=axes[0, 1], ticks=[0, 0.2, 0.4, 0.6, 0.8, 1]
    )
    plt.colorbar(
        rho_plot, label=r"$\rho$", ax=axes[0, 0], ticks=[0, 0.2, 0.4, 0.6, 0.8, 1]
    )

    vmin = np.min(v.T)
    vmax = np.max(v.T)

    # v_comp = v_comp.at[v_comp > vmax].set(vmax)
    # v_comp = v_comp.at[v_comp < vmin].set(vmin)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # plot v
    v_plot = axes[1, 0].contourf(
        t_mesh, x_mesh, v[1:-3].T, levels=100, cmap=cmap, norm=norm
    )
    v_computed_plot = axes[1, 1].contourf(
        t_mesh, x_mesh, v_comp[1:-3].T, levels=100, cmap=cmap, norm=norm
    )
    plt.colorbar(v_plot, ax=axes[1, 1], label=r"$v$")
    plt.colorbar(v_plot, ax=axes[1, 0], label=r"$v$")

    # plot f

    vmin = np.min(f.T)
    vmax = np.max(f.T)

    # f_comp = f_comp.at[f_comp > vmax].set(vmax)
    # f_comp = f_comp.at[f_comp < vmin].set(vmin)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    f_plot = axes[2, 0].contourf(
        t_mesh, x_mesh, f[1:-3].T, levels=100, cmap=cmap, norm=norm
    )
    f_computed_plot = axes[2, 1].contourf(
        t_mesh, x_mesh, f_comp[1:-3].T, levels=100, cmap=cmap, norm=norm
    )
    plt.colorbar(f_plot, ax=axes[2, 1], label=r"$f$")
    plt.colorbar(f_plot, ax=axes[2, 0], label=r"$f$")

    for i in range(3):
        axes[i, 0].set_xlabel(r"t")
        axes[i, 1].set_xlabel(r"t")
        axes[i, 0].set_ylabel(r"x")
        axes[i, 1].set_ylabel(r"x")
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.savefig("plots.png", dpi=300)
