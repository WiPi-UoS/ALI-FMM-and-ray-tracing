import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # True to save plots to local directory, False to view
    save_fig = False

    # Load geometry
    veln = np.load("weld_veln.npy")
    velpn = np.load("weld_velpn.npy")
    dnx = 0.0002
    [nnz, nnx] = veln.shape
    plot_veln = np.copy(veln) % 90
    for i in range(veln.shape[0]):
        for j in range(veln.shape[1]):
            if velpn[i, j] == 1:
                plot_veln[i, j] = None

    # Load ray paths (note, ray paths are not scaled by dnx i.e use grid coordinates)
    ray_paths_x = np.load("ray_paths_x.npy")
    ray_paths_y = np.load("ray_paths_y.npy")
    ray_len = np.load("ray_len.npy")

    for i in range(ray_len.shape[0]):
        # Check if any ray paths exist for source
        if sum(ray_len[i, :]) != 0:
            # Plot geometry
            plt.imshow(plot_veln, vmin=0, vmax=90, cmap="hsv", interpolation="nearest", extent=[0, dnx * (nnx - 1), dnx * (nnz - 1), 0])
            plt.colorbar(label="Orienation")
            plt.gca().invert_yaxis()
            for j in range(ray_len.shape[0]):
                # if ray path exists
                if ray_len[i, j] != 0:
                    plt.plot(dnx * ray_paths_x[i, j, 0:ray_len[i, j]], dnx * ray_paths_y[i, j, 0:ray_len[i, j]], "k", linewidth=1)
            if save_fig:
                # Save figure and clear plot
                plt.savefig(f"Ray_paths_{i}.png")
                plt.close()
            else:
                # View figure
                plt.show()
