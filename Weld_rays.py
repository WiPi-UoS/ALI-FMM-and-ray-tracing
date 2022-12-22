import numpy as np
from Anis_TTF_rays import ALI_FMM
import matplotlib.pyplot as plt

if __name__ == "__main__":
    n_threads = 8

    # Load material properties
    veln = np.load("weld_veln.npy")
    velpn = np.load("weld_velpn.npy").astype(int)
    vel_map = np.load("weld_vel_map.npy")
    stif_density = np.load("weld_stif_den.npy")

    # Set up grid spacing and transducer positioning
    dnx = 0.0002
    [nnz, nnx] = veln.shape
    n_trans = 31
    gap_len = 15
    center = nnx / 2
    trans_len = gap_len * (n_trans - 1)
    start_x = center - trans_len / 2
    end_x = center + trans_len / 2
    source_x = dnx * np.arange(start_x, end_x + gap_len / 2, gap_len)
    source_y = dnx * np.array([0, nnz - 1])

    source_x1 = []
    source_y1 = []
    for i in range(n_trans):
        source_x1.append(source_x[i])
        source_y1.append(source_y[0])
    for i in range(n_trans):
        source_x1.append(source_x[i])
        source_y1.append(source_y[1])
    source_x1 = np.array(source_x1)
    source_y1 = np.array(source_y1)

    # Set up orientations for plotting with no orientation for isotropic material
    plot_veln = np.copy(veln) % 90
    for i in range(veln.shape[0]):
        for j in range(veln.shape[1]):
            # Since no velocity curve is being given 0 corresponds to using stifness tensors and density and 1 corresponds to an isotropic medium
            if velpn[i, j] == 1:
                plot_veln[i, j] = None

    # Plot geometry and transducer positions
    plt.imshow(plot_veln, vmin=0, vmax=90, cmap="hsv", interpolation="nearest", extent=[0, dnx * (nnx - 1), dnx * (nnz - 1), 0])
    plt.gca().invert_yaxis()
    plt.plot(source_x1, source_y1, "kx")
    plt.show()

    # Define which transducer pairs are being used for ray tracing.Ray paths are only calculated between top and bottom source/receivers
    trans_pairs = np.zeros((2 * n_trans, 2 * n_trans))
    for i in range(n_trans):
        for j in range(n_trans, 2 * n_trans):
            trans_pairs[i, j] = 1

    # Initiate class
    Forward_Model = ALI_FMM(veln, velpn, vel_map, source_x1, source_y1, stif_den=stif_density, dnx=dnx)

    # Calculate ray paths in parallel
    trav_times = Forward_Model.find_all_TTF_rays_parallel(veln, velpn, vel_map, stif_den=stif_density, n_threads=n_threads, trans_pairs=trans_pairs)

    # Shorten arrays length to size of the longest ray path
    max_len = np.max(Forward_Model.ray_len)
    ray_paths_x = Forward_Model.ray_paths_x[:, :, 0:max_len]
    ray_paths_y = Forward_Model.ray_paths_y[:, :, 0:max_len]

    # Save travel times and ray paths so they can be plotted using plot_rays.py
    np.save("trav_times.npy", trav_times)
    np.save("ray_paths_x.npy", ray_paths_x)
    np.save("ray_paths_y.npy", ray_paths_y)
    np.save("ray_len.npy", Forward_Model.ray_len)
