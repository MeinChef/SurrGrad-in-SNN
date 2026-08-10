from imports import MaxNLocator, plt, torch
from imports import numpy as np
from imports import snntorch as snn
from imports import spikeplot as splt


def plot_lif_voltage() -> None:
    """
    Adapted from https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html
    """
    neuron = snn.Leaky(0.9)

    # Initialize inputs and outputs
    cur_in = torch.cat((torch.zeros(10, 1), torch.ones(50, 1)*0.15, torch.zeros(20, 1)), 0)
    mem = torch.zeros(1)
    spk_out = torch.zeros(1)
    mem_rec = [mem]
    spk_rec = [spk_out]

    # Simulation run across 100 time steps.
    for step in range(cur_in.shape[0]):
        spk_out, mem = neuron(cur_in[step], mem)
        mem_rec.append(mem)
        spk_rec.append(spk_out)

    # convert lists to tensors
    mem_rec = torch.stack(mem_rec)
    spk_rec = torch.stack(spk_rec)


    # def plot_cur_mem_spk(cur, mem, spk, thr_line=False, vline=False, title=False, ylim_max2=1.25):
    # Generate Plots
    fig, ax = plt.subplots(
        4, 
        figsize = (10,8), 
        sharex = False, 
        gridspec_kw = {'height_ratios': [1, 1, 0.4, 0.7]}
    )

    # Plot input current
    ax[0].plot(cur_in, c = "tab:orange")
    ax[0].set_ylim([0, 0.4])
    ax[0].set_xlim([0, cur_in.shape[0]])
    ax[0].set_ylabel("Input Current ($I_{in}$)")
    ax[0].set_title("Leaky Integrate-and-Fire Neuron with step input")

    # Plot membrane potential
    ax[1].plot(mem_rec)
    ax[1].set_ylim([0, 1.3]) 
    ax[1].set_ylabel("Membrane Potential ($U_{mem}$)")
    ax[1].axhline(
        y = 1, 
        alpha = 0.25, 
        linestyle = "dashed", 
        c = "black", 
        linewidth = 2
    )
    ax[1].sharex(ax[0])

    # Plot output spike using spikeplot
    splt.raster(
        spk_rec, 
        ax[2], 
        s = 400, 
        c = "black", 
        marker = "|"
    )
    ax[2].sharex(ax[0])

    ax[2].set_xlabel("Time step")
    ax[2].set_ylabel("Output spikes")
    ax[2].set_yticks([]) 

    # ------------------------------------------------------------------
    # Zoomed spike train with binary representation
    # ------------------------------------------------------------------
    start, end = 20, 40

    zoom_spikes = spk_rec[start:end].squeeze().int().numpy()
    times = np.arange(start, end, dtype = np.int32)

    # Show spikes
    ax[3].vlines(
        times[zoom_spikes == 1],
        0,
        1,
        color="black",
        linewidth=2,
    )

    ax[3].set_xlim(start - 0.5, end - 0.5)
    ax[3].set_ylim(-0.8, 1.2)
    ax[3].set_yticks([])
    ax[3].set_xlabel("Time step")
    ax[3].set_title(f"Spike train ({start}–{end}) Array representation")

    # Binary values underneath each timestep
    for t, val in zip(times, zoom_spikes):
        ax[3].text(
            t,
            -0.35,
            str(val),
            ha = "center",
            va = "center",
            fontsize = 11,
            family = "monospace",
        )
    ax[3].xaxis.set_major_locator(
        MaxNLocator(integer = True)
    )

    fig.tight_layout()
    plt.savefig("./img/lif-neuron.pdf", format = "pdf")
    plt.show()

if __name__ == "__main__":
    plot_lif_voltage()