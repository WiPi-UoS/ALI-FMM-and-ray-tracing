"""
Anisotropic traveltime fields and ray tracing module
__________________________________________________________________________________

This module uses ALI-FMM for modeling traveltime fields and ray paths in anisotropic media.

The required python packages are multiprocessing, matplotlib, numpy, math, numba and tqdm.

Class should not be used without  if __name__ == '__main__':  as this may prevent codes from running correctly.
"""

import time
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import math
from numba import njit, jit
import numba
from time import sleep
from tqdm.auto import tqdm

# Parameter used to enable/disable progress bars
#tqdm_disable = True
tqdm_disable = False

@njit(cache=True)
def finer_grid_n(veln, scale, dtype=numba.int32):
    """
    Function for creating a finer grid for a 2D array. Values at each point is the same as the closest point in the original grid.

    :param veln: Array that finer grid is created from
    :type veln: 2D numpy array
    :param scale: How much the grid size is increased by in each direction.
    :type scale: Odd integer
    :param dtype: Data type of the new array. Since function uses numba njit, dtype must be from numba. Default is numba.int32
    :type dtype: data type
    :return: 2D numpy array: Finer grid.
    :rtype: 2D numpy array of type dtype
    """
    # Determine the new dimensions of the new grid and create array
    dim = veln.shape
    new_dim = (scale * (dim[0] - 1) + 1, scale * (dim[1] - 1) + 1)
    new_veln = np.zeros(new_dim, dtype=dtype)
    #new_veln = np.zeros(new_dim, dtype=int)

    # For each point in the original grid assign all surrounding points closest to that point in the new grid are assigned with the same value.
    side = int((scale - 1) / 2)
    for i in range(veln.shape[0]):
        for j in range(veln.shape[1]):
            left = max(0, scale * i - side)
            right = min(scale * i + side, new_dim[0] - 1)
            bottom = max(0, scale * j - side)
            top = min(scale * j + side, new_dim[1] - 1)
            new_veln[left:right + 1, bottom:top + 1] = veln[i, j] * np.ones((right - left + 1, top - bottom + 1), dtype=dtype)
            #new_veln[left:right + 1, bottom:top + 1] = veln[i, j] * np.ones((right - left + 1, top - bottom + 1), dtype=int)
    return new_veln


@njit(cache=True)
def finer_grid_n_2(data, scale):
    """
    Function for creating a finer grid from a 3D array of material parameters (c_22, c_23, c_33, c_44, density). If None is input None is returned.

    :param data: Material parameters with first two indices being the i,j coordinates and the third being the index of the material parameter in order (c_22, c_23, c_33, c_44, density)
    :type data: 3D numpy array of type int64
    :param scale: How much the grid size is increased by.
    :type scale: Odd integer
    :return: Material parameters at all points in the finer grid. None if data=None
    :rtype: 3D numpy array of type int64
    """
    # if None is input None is returned.
    if data == None:
        return None
    else:
        # Determine the new dimensions of the new grid and create array
        dim = data.shape
        new_dim = (scale * (dim[0] - 1) + 1, scale * (dim[1] - 1) + 1)
        new_data = np.zeros((new_dim[0], new_dim[1], dim[2]), dtype=numba.int64)
        #new_veln = np.zeros(new_dim, dtype=int)

        # For each point in the original grid assign all surrounding points closest to that point in the new grid are assigned with the same material parameters.
        side = int((scale - 1) / 2)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                left = max(0, scale * i - side)
                right = min(scale * i + side, new_dim[0] - 1)
                bottom = max(0, scale * j - side)
                top = min(scale * j + side, new_dim[1] - 1)
                for k in range(data.shape[2]):
                    new_data[left:right + 1, bottom:top + 1, k] = data[i, j, k] * np.ones((right - left + 1, top - bottom + 1), dtype=numba.int64)
        return new_data


@njit(cache=True)
def addtree(iz, ix, nsts, btg, ntr, ttn):
    """
    Function for adding a new value to the minimum heap.

    :param iz: z index of point being added to heap.
    :type iz: int
    :param ix: x index of point being added to heap.
    :type ix: int
    :param nsts: Note status for points in the array. -1 is for unknown point, if point is still in the heap then value is the position in the tree.
    :type nsts: 2D numpy array of type int
    :param btg: Array for storing the positions of points in the min heap.
    :type btg: 2D numpy array of type int
    :param ntr: Number of points on the min heap.
    :type ntr: int
    :param ttn: Current travel times at all points in the grid (0 for "far" points).
    :type ttn: 2D numpy array
    :return: nsts, btg, ntr - Updated values of the parameters nsts, btg and ntr
    :rtype: 2D numpy array of type int, 2D numpy array of type int, int
    """
    # Increase the size of the tree by one.
    ntr = ntr + 1
    # Put new value at base of tree
    nsts[iz, ix] = ntr
    btg[ntr, 1] = ix
    btg[ntr, 0] = iz

    # Now filter the new value up to its correct position
    tpc = ntr  # Tree position of parent node.
    tpp = round(tpc / 2)  # Tree position of child node.
    while tpp > 0:
        # Get i, j coordinates of parent node
        aa = btg[tpp, 0]
        bb = btg[tpp, 1]
        if ttn[iz, ix] < ttn[aa, bb]:  # Check whether to swap new point with the parent node/
            nsts[iz, ix] = tpp
            nsts[btg[tpp, 0], btg[tpp, 1]] = tpc
            exch = np.copy(btg[tpc, :])  # temporary value for swapping values
            btg[tpc, :] = btg[tpp, :]
            btg[tpp, :] = exch
            tpc = tpp  # Set position of new point to parents (swapped).
            tpp = round(tpc / 2)  # Find position in binary tree of new parent
        else:
            tpp = 0
    return nsts, btg, ntr


@njit(cache=True)
def updtree(iz, ix, nsts, btg, ttn):
    """
    Updates the value of a point in a min heap and and filters value to new position in binary tree.

    :param iz: z index of point being updated.
    :type iz: int
    :param ix: x index of point being updated.
    :type iz: int
    :param nsts: Node status for points in the array. -1 is for unknown point, if point is still in the heap then value is the position in the tree.
    :type nsts: 2D numpy array of type int
    :param btg: Positions of points in the min heap.
    :type btg: 2D numpy array of type int
    :param ttn: Current travel time at all points in the grid (0 for far points).
    :type ttn: 2D numpy array
    :return: nsts, btg - Updated values of parameters nsts and btg.
    :rtype: 2D numpy array of type int, 2D numpy array of type int
    """
    tpc = 1 * nsts[iz, ix]  # Position in the binary tree of the point being updated.
    tpp = round(tpc / 2)  # Position in the binary tree of the parent node.

    # Filter point to new position in binary tree.
    while tpp > 0:
        if ttn[iz, ix] < ttn[btg[tpp, 0], btg[tpp, 1]]:  # Check if child and parent nodes should be swapped.
            # Swap nodes
            nsts[iz, ix] = tpp
            nsts[btg[tpp, 0], btg[tpp, 1]] = tpc
            exch = np.copy(btg[tpc, :])  # Temp value for swapping nodes.
            btg[tpc, :] = btg[tpp, :]
            btg[tpp, :] = exch
            tpc = tpp
            tpp = round(tpc / 2)
        else:
            tpp = 0
    return nsts, btg


@njit(cache=True)
def downtree(nsts, btg, ntr, ttn):
    """
    Function for removing the root of the min heap and filtering points into correct positions. Root is replaced by the point at the bottom of the tree and then filtered down.

    :param nsts: Node status for points in the array. -1 is for unknown point, if point is still in the heap then value is the position in the tree.
    :type nsts: 2D numpy array of type int
    :param btg: Positions of points in the min heap.
    :type btg: 2D numpy array of type int
    :param ntr: Number of points in binary tree.
    :type ntr: int
    :param ttn: Current travel time at all points in the grid (0 for far points).
    :type ttn: 2D numpy array
    :return: nsts, btg, ntr, ttn - Updated values of parameters nsts, btg, ntr and ttn.
    :rtype: 2D numpy array of type int, 2D numpy array of type int, int, 2D numpy array
    """
    if ntr == 1:  # If removing last point in the tree.
        ntr = ntr - 1
        return nsts, btg, ntr, ttn

    # Replace root of tree with the point in the last position and decrease tree size.
    nsts[btg[ntr, 0], btg[ntr, 1]] = 1
    btg[1, :] = np.copy(btg[ntr, :])
    ntr = ntr - 1

    # Filter new root down to its correct position.
    tpp = 1  # Position in binary tree of parent node.
    tpc = 2 * tpp  # Position in binary tree of parent node.
    while tpc < ntr:
        # Find smallest value of the two child nodes.
        rd1 = ttn[btg[tpc, 0], btg[tpc, 1]]
        rd2 = ttn[btg[tpc + 1, 0], btg[tpc + 1, 1]]
        if rd1 > rd2:
            tpc = tpc + 1

        # Check whether the child is smaller than the parent; if so, then swap, if not, then we are done.
        rd1 = ttn[btg[tpc, 0], btg[tpc, 1]]
        rd2 = ttn[btg[tpp, 0], btg[tpp, 1]]
        if rd1 < rd2:
            nsts[btg[tpp, 0], btg[tpp, 1]] = tpc
            nsts[btg[tpc, 0], btg[tpc, 1]] = tpp
            exch = np.copy(btg[tpc, :])
            btg[tpc, :] = btg[tpp, :]
            btg[tpp, :] = exch
            tpp = tpc
            tpc = 2 * tpp
        else:
            tpc = ntr + 1

    # If ntr is an even number, then we still have one more test to do.
    if tpc == ntr:
        rd1 = ttn[btg[tpc, 0], btg[tpc, 1]]
        rd2 = ttn[btg[tpp, 0], btg[tpp, 1]]
        if rd1 < rd2:
            nsts[btg[tpp, 0], btg[tpp, 1]] = tpc
            nsts[btg[tpc, 0], btg[tpc, 1]] = tpp
            exch = np.copy(btg[tpc, :])
            btg[tpc, :] = btg[tpp, :]
            btg[tpp, :] = exch
    return nsts, btg, ntr, ttn


@njit(cache=True)
def fouds18_A(iz, ix, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, vel_map, avlist2, stif_den):
    """
    Finite difference method for calculating first arrival travel time at iz, ix from travel times of surrounding points with travel time estimates (not all points have an assigned travel time. Uses finite difference method from Anisotropic Multi-Stencil Fast Marching Method (from tant et al 2020 (Effective grain orientation mapping ....)).

    :param iz: z index of point being updated.
    :type iz: int
    :param ix: x index of point being updated.
    :type iz: int
    :param nsts: Node status for points in the array. -1 is for unknown point, if point is still in the heap then value is the position in the tree.
    :type nsts: 2D numpy array of type int
    :param ttn: Current travel time at all points in the grid (0 for far points).
    :type ttn: 2D numpy array
    :param dnx: Distance between points in the grid in the x direction.
    :type dnx: float
    :param dnz: Distance between points in the grid in the z direction.
    :type dnz: float
    :param nnx: Number of points in the grid in the x direction.
    :type nnx: int
    :param nnz: Number of points in the grid in the z direction.
    :type nnz: int
    :param veln: Anisotropic orientation of all grid points.
    :type veln: 2D numpy array
    :param velpn: Material index of all grid points (0 if using stiffness tensors and density, otherwise index for column in avlist2).
    :type velpn: 2D numpy array of type int
    :param vel_map: Values used for scaling velocities at all grid points (mainly used for isotropic materials).
    :type vel_map: 2D numpy array
    :param avlist2: Group velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type avlist2: 2D numpy array
    :param stif_den: Material parameters with first two indices being the i,j coordinates and the third being the index of the material parameter(c_22, c_23, c_33, c_44, density). Stiffness tensors must be in MPa to avoid overflow errors.
    :type stif_den: 3D numpy array of type np.int64
    :return: Travel time estimate at the point iz, ix.
    :rtype: float
    """


    # ! Inspect each of the four quadrants for the minimum time
    # ! solution.

    # CALCULATE 2ND ORDER TRAVEL TIMES ON 0DEG STENCIL

    tsw1 = 0
    travm = 0
    wave_ang = 0  # KT 31 / 5 / 17 calculate ray angle between neighbouring alive point and node for determination
    # eff_ang = round(mod(wave_ang - veln(iz, ix), 90)) # !KT 31 / 15 / 17 effective angle by subtracting cell
    # orientation from ray angle
    eff_ang = (wave_ang - veln[iz, ix]) % 180
    if velpn[iz, ix] != 0 or stif_den == None:
        angle1 = math.floor(eff_ang)
        angle2 = (angle1 + 1) % 180
        remainder = eff_ang - angle1
        velocity = vel_map[iz, ix] * ((1 - remainder) * avlist2[angle1, velpn[iz, ix]] + remainder * avlist2[angle2, velpn[iz, ix]])
    else:
        # Solves christoffel equation to find group velocity.
        sigma = stif_den[iz, ix, 4]
        if eff_ang % 90 < 0.01 or eff_ang % 90 > 90 - 0.01:
            if abs((eff_ang % 180) - 90) < 1:
                lambda_val = stif_den[iz, ix, 2]
            else:
                lambda_val = stif_den[iz, ix, 0]
            velocity = 1000 * vel_map[iz, ix] * math.sqrt(lambda_val / sigma)
        else:
            c_22 = stif_den[iz, ix, 0]
            c_23 = stif_den[iz, ix, 1]
            c_33 = stif_den[iz, ix, 2]
            c_44 = stif_den[iz, ix, 3]
            tan_ang = math.tan(math.radians(eff_ang))
            A = c_22 + c_33 - 2 * c_44
            B = (c_23 + c_44) * (tan_ang - 1 / tan_ang)
            C = c_22 - c_33
            if eff_ang < 90:
                phase_angle_rad = math.atan((-B - math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
            else:
                phase_angle_rad = math.atan((-B + math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
            lambda_val = 0.5 * (math.cos(2 * phase_angle_rad) * (c_22 - c_44) + math.sin(2 * phase_angle_rad) * (c_23 + c_44) * tan_ang + c_22 + c_44)
            velocity = 1000 * vel_map[iz, ix] * math.sqrt(lambda_val / sigma) / math.cos(math.radians(eff_ang) - phase_angle_rad)

    slown = 1.0 / velocity

    for j in [ix - 1, ix + 1]:  # j=ix-1:2: ix + 1
        if 0 <= j <= nnx - 1:
            swj = -1
            if j == ix - 1:
                j2 = j - 1
                if j2 >= 0:
                    if nsts[iz, j2] == 0:
                        swj = 0
            else:
                j2 = j + 1
                if j2 <= nnx - 1:
                    if nsts[iz, j2] == 0:
                        swj = 0
            if nsts[iz, j] == 0 and swj == 0:
                swj = -1
                if ttn[iz, j] >= ttn[iz, j2]:
                    swj = 0
            else:
                swj = -1
            for k in [iz - 1, iz + 1]:  # k=iz-1:2: iz + 1
                if 0 <= k <= nnz - 1:
                    swk = -1
                    if k == iz - 1:
                        k2 = k - 1
                        if k2 >= 0:
                            if nsts[k2, ix] == 0:
                                swk = 0
                    else:
                        k2 = k + 1
                        if k2 <= nnz - 1:
                            if nsts[k2, ix] == 0:
                                swk = 0
                    if nsts[k, ix] == 0 and swk == 0:
                        swk = -1
                        if ttn[k, ix] >= ttn[k2, ix]:
                            swk = 0
                    else:
                        swk = -1
                    swsol = 0

                    if swj == 0:
                        swsol = 1
                        if swk == 0:
                            u = 2.0 * dnx
                            a = 18
                            b = -6 * (4.0 * ttn[iz, j] - ttn[iz, j2] + 4.0 * ttn[k, ix] - ttn[k2, ix])
                            c = (4.0 * ttn[iz, j] - ttn[iz, j2]) ** 2 + (
                                    4.0 * ttn[k, ix] - ttn[k2, ix]) ** 2 - 4 * u ** 2 * slown ** 2
                            tref = 0.0
                            tdiv = 1.0
                        elif nsts[k, ix] == 0:
                            u = dnz
                            v = 2.0 * dnx
                            a = 18
                            b = -6.0 * (3.0 * ttn[k, ix] + 4.0 * ttn[iz, j] - ttn[iz, j2])
                            c = (3.0 * ttn[k, ix]) ** 2 + (
                                    4.0 * ttn[iz, j] - ttn[iz, j2]) ** 2 - 4 * v ** 2 * slown ** 2
                            tref = 0.0
                            tdiv = 1.0 #############################
                            #v = dnx
                            #a = 13
                            #b = - 8.0 * ttn[k, ix] - 6.0 * (4.0 * ttn[iz, j] - ttn[iz, j2])
                            #c = 4.0 * ttn[k, ix] ** 2 + (
                            #            4.0 * ttn[iz, j] - ttn[iz, j2]) ** 2 - 4 * v ** 2 * slown ** 2
                        else:
                            u = 2.0 * dnx
                            a = 1.0
                            b = 0.0
                            c = -u ** 2 * slown ** 2
                            tref = 4.0 * ttn[iz, j] - ttn[iz, j2]
                            tdiv = 3.0
                            #u = dnx
                            #a = 9.0
                            #b = 0.0
                            #c = -1 * (4.0 * ttn[iz, j] - ttn[iz, j2]) ** 2 + 3 * (u * slown) ** 2
                            #tref = 0.0  # 4.0 * ttn[iz, j] - ttn[iz, j2]
                            tdiv = 1.0
                    elif nsts[iz, j] == 0:
                        swsol = 1
                        if swk == 0:
                            u = dnx
                            em = 3.0 * ttn[iz, j] + 4.0 * ttn[k, ix] - ttn[k2, ix]
                            a = 18
                            b = -6.0 * em
                            c = (3.0 * ttn[iz, j]) ** 2 + (
                                        4.0 * ttn[k, ix] - ttn[k2, ix]) ** 2 - 3 * 4 * u ** 2 * slown ** 2
                            tref = 0.0
                            tdiv = 1.0 ##########################################
                            #u = dnx
                            #a = 13.0
                            #b = - 6.0 * (4.0 * ttn[k, ix] - ttn[k2, ix]) - 8.0 * 3.0 * ttn[iz, j]
                            #c = (4.0 * ttn[k, ix] - ttn[k2, ix]) ** 2 + 4.0 * ttn[iz, j] ** 2 - 4 * (u * slown) ** 2
                            #tref = 0.0
                            #tdiv = 1.0
                        elif nsts[k, ix] == 0:
                            u = dnx
                            v = dnz
                            a = 2
                            b = -2 * (ttn[k, ix] + ttn[iz, j])
                            c = ttn[k, ix] ** 2 + ttn[iz, j] ** 2 - (u * slown) ** 2
                            tref = 0.0
                            tdiv = 1.0
                        else:
                            a = 1.0
                            b = 0.0
                            c = -(ttn[iz, j] + slown * dnx) ** 2
                            tref = 0.0
                            tdiv = 1.0
                    else:
                        if swk == 0:
                            swsol = 1
                            u = 2.0 * dnz
                            a = 1.0
                            b = 0.0
                            c = -u ** 2 * slown ** 2
                            tref = 4.0 * ttn[k, ix] - ttn[k2, ix]
                            tdiv = 3.0 #########################################
                            #a = 9.0
                            #b = 0.0
                            #c = -(4.0 * ttn[k, ix] - ttn[k2, ix] + (u * slown)) ** 2
                            #tref = 0.0
                            #tdiv = 1.0
                        elif nsts[k, ix] == 0:
                            swsol = 1
                            a = 1.0
                            b = 0.0
                            c = -(ttn[k, ix] + slown * dnz) ** 2
                            tref = 0.0
                            tdiv = 1.0
                    # Now find the solution of the quadratic equation
                    if swsol == 1:
                        rd1 = b ** 2 - 4.0 * a * c
                        if rd1 < 0:
                            rd1 = 0
                        tdsh = (-b + math.sqrt(rd1)) / (2.0 * a)
                        trav = (tref + tdsh) / tdiv
                        if tsw1 == 1:
                            travm = min(trav, travm)
                        else:
                            travm = trav
                            tsw1 = 1
    #
    # if travm~=0
    # ttn(iz, ix) = min(travm, trav);
    # else
    # ttn(iz, ix) = trav;
    # end

    tsw2 = 0
    travmd = 0
    wave_ang = 45  # KT 31 / 5 / 17 calculate ray angle between neighbouring alive point and node for determination
    eff_ang = round((wave_ang - veln[iz, ix]) % 180)  # round(mod(wave_ang - veln(iz, ix), 90)) # !KT 31 / 15 / 17
    # effective angle by subtracting cell orientation from ray angle
    if velpn[iz, ix] != 0 or stif_den == None:
        angle1 = math.floor(eff_ang)
        angle2 = (angle1 + 1) % 180
        remainder = eff_ang - angle1
        velocity = vel_map[iz, ix] * ((1 - remainder) * avlist2[angle1, velpn[iz, ix]] + remainder * avlist2[angle2, velpn[iz, ix]])
    else:
        # Solves christoffel equation to find group velocity.
        sigma = stif_den[iz, ix, 4]
        if eff_ang % 90 < 0.01 or eff_ang % 90 > 90 - 0.01:
            if abs((eff_ang % 180) - 90) < 1:
                lambda_val = stif_den[iz, ix, 2]
            else:
                lambda_val = stif_den[iz, ix, 0]
            velocity = 1000 * vel_map[iz, ix] * math.sqrt(lambda_val / sigma)
        else:
            c_22 = stif_den[iz, ix, 0]
            c_23 = stif_den[iz, ix, 1]
            c_33 = stif_den[iz, ix, 2]
            c_44 = stif_den[iz, ix, 3]
            tan_ang = math.tan(math.radians(eff_ang))
            A = c_22 + c_33 - 2 * c_44
            B = (c_23 + c_44) * (tan_ang - 1 / tan_ang)
            C = c_22 - c_33
            if eff_ang < 90:
                phase_angle_rad = math.atan((-B - math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
            else:
                phase_angle_rad = math.atan((-B + math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
            lambda_val = 0.5 * (math.cos(2 * phase_angle_rad) * (c_22 - c_44) + math.sin(2 * phase_angle_rad) * (c_23 + c_44) * tan_ang + c_22 + c_44)
            velocity = 1000 * vel_map[iz, ix] * math.sqrt(lambda_val / sigma) / math.cos(math.radians(eff_ang) - phase_angle_rad)

    slown = 1.0/velocity
    mf2 = math.sqrt(2)  # distance between two diagonally connected points

    # start with diag axes of stencil
    for j in [ix - 1, ix + 1]:  # j=ix-1:2: ix + 1
        if j == ix - 1:
            k = iz + 1
        else:
            k = iz - 1
        if 0 <= j <= nnx - 1 and 0 <= k <= nnz - 1:
            swdiag = -1
            if j == ix - 1:
                j2 = j - 1
                k2 = k + 1
                if j2 >= 0 and k2 <= nnz - 1:
                    if nsts[k2, j2] == 0:
                        swdiag = 0
            elif j == ix + 1:
                j2 = j + 1
                k2 = k - 1
                if j2 <= nnx - 1 and k2 >= 0:
                    if nsts[k2, j2] == 0:
                        swdiag = 0
            if nsts[k, j] == 0 and swdiag == 0:
                swdiag = -1
                if ttn[k, j] >= ttn[k2, j2]:
                    swdiag = 0
            else:
                swdiag = -1

            # skew diag axis
            for jj in [ix - 1, ix + 1]:  # jj=ix-1:2: ix + 1
                if jj == ix - 1:
                    kk = iz - 1
                else:
                    kk = iz + 1

                if 0 <= jj <= nnx - 1 and 0 <= kk <= nnz - 1:
                    swskew = -1
                    if jj == ix - 1:
                        jj2 = jj - 1
                        kk2 = kk - 1
                        if jj2 >= 0 and kk2 >= 0:
                            if nsts[kk2, jj2] == 0:
                                swskew = 0
                    elif jj == ix + 1:
                        jj2 = jj + 1
                        kk2 = kk + 1
                        if jj2 <= nnx - 1 and kk2 <= nnz - 1:
                            if nsts[kk2, jj2] == 0:
                                swskew = 0
                    if nsts[kk, jj] == 0 and swskew == 0:
                        swskew = -1
                        if ttn[kk, jj] >= ttn[kk2, jj2]:
                            swskew = 0
                    else:
                        swskew = -1
                    swsol = 0

                    if swdiag == 0:
                        swsol = 1
                        if swskew == 0:
                            u = 2.0 * mf2 * dnx
                            a = 18.0
                            b = -6.0 * (4.0 * ttn[k, j] - ttn[k2, j2] + 4.0 * ttn[kk, jj] - ttn[kk2, jj2])
                            c = (4.0 * ttn[k, j] - ttn[k2, j2]) ** 2 + (
                                        4.0 * ttn[kk, jj] - ttn[kk2, jj2]) ** 2 - 4 * u ** 2 * slown ** 2
                            tref = 0
                            tdiv = 1.0
                            #u = mf2 * dnx
                            #a = 18.0
                            #b = -6.0 * (4.0 * ttn[k, j] - ttn[k2, j2] + 4.0 * ttn[kk, jj] - ttn[kk2, jj2])
                            #c = (4.0 * ttn[k, j] - ttn[k2, j2]) ** 2 + (
                            #        4.0 * ttn[kk, jj] - ttn[kk2, jj2]) ** 2 - 4 * u ** 2 * slown ** 2
                            #tref = 0
                            #tdiv = 1.0
                        elif nsts[kk, jj] == 0:
                            #v = mf2 * dnx          ###wrong index
                            #a = 45
                            #b = -6.0 * (12.0 * ttn[k, ix] + 4.0 * ttn[iz, j] - ttn[iz, j2])
                            #c = (6.0 * ttn[k, ix]) ** 2 + (
                            #        4.0 * ttn[iz, j] - ttn[iz, j2]) ** 2 - 16 * v ** 2 * slown ** 2
                            #tref = 0.0
                            #tdiv = 1.0
                            u = mf2 * dnz
                            v = 2.0 * mf2 * dnx
                            a = 18
                            b = -6.0 * (3.0 * ttn[kk, jj] + 4.0 * ttn[k, j] - ttn[k2, j2])
                            c = (3.0 * ttn[kk, jj]) ** 2 + (
                                    4.0 * ttn[k, j] - ttn[k2, j2]) ** 2 - 4 * v ** 2 * slown ** 2
                            tref = 0.0
                            tdiv = 1.0
                            #v = mf2 * dnx
                            #a = 13
                            #b = -6.0 * (4.0 * ttn[iz, j] - ttn[iz, j2]) - 8.0 * ttn[k, ix]
                            #c = 4.0 * ttn[k, ix] ** 2 + (4.0 * ttn[iz, j] - ttn[iz, j2]) ** 2 - 4.0 * (v * slown) ** 2
                            #tref = 0.0
                            #tdiv = 1.0
                        else:
                            u = mf2 * 2.0 * dnx
                            a = 1.0
                            b = 0.0
                            c = - 1.0 * (u * slown) ** 2
                            tref = (4.0 * ttn[k, j] - ttn[k2, j2])
                            tdiv = 3.0
                            #u = mf2 * dnx
                            #a = 9.0
                            #b = 0.0
                            #c = -(4.0 * ttn[k, j] - ttn[k2, j2]) ** 2 + 3 * (u * slown) ** 2
                            #tref = 0.0
                            #tdiv = 1.0
                    elif nsts[k, j] == 0:
                        swsol = 1
                        if swskew == 0:
                            u = mf2 * dnx
                            v = mf2 * 2.0 * dnz
                            em = 3.0 * ttn[k, j] + 4.0 * ttn[kk, jj] - ttn[kk2, jj2]
                            a = 18
                            b = -6.0 * em
                            c = (3.0 * ttn[k, j]) ** 2 + (4.0 * ttn[kk, jj] - ttn[kk2, jj2]) ** 2 - 3 * 4 * u ** 2 * slown ** 2
                            tref = 0.0
                            tdiv = 1.0
                            #u = mf2 * dnx
                            #a = 13
                            #b = -6.0 * (4.0 * ttn[kk, jj] - ttn[kk2, jj2]) - 8.0 * ttn[k, j]
                            #c = (4.0 * ttn[kk, jj] - ttn[kk2, jj2]) ** 2 + 4.0 * ttn[k, j] ** 2 - 4.0 * (u * slown) ** 2
                            #tref = 0.0
                            #tdiv = 1.0
                        elif nsts[kk, jj] == 0:
                            u = mf2 * dnx
                            v = mf2 * dnz
                            a = 2
                            b = -2 * (ttn[kk, jj] + ttn[k, j])
                            c = ttn[kk, jj] ** 2 + ttn[k, j] ** 2 - 4 / 9 * (u * slown) ** 2
                            tref = 0.0
                            tdiv = 1.0
                            #u = mf2 * dnx
                            #a = 2
                            #b = -2 * (ttn[kk, jj] + ttn[k, j])
                            #c = ttn[kk, jj] ** 2 + ttn[k, j] ** 2 - (u * slown) ** 2
                            #tref = 0.0
                            #tdiv = 1.0
                        else:
                            u = mf2 * dnx
                            a = 1.0
                            b = 0.0
                            c = -(ttn[k, j] + slown * u) ** 2
                            tref = 0
                            tdiv = 1.0
                    else:
                        if swskew == 0:
                            swsol = 1
                            u = 2.0 * mf2 * dnz
                            a = 1.0
                            b = 0.0
                            c = -u ** 2 * slown ** 2
                            tref = 4.0 * ttn[kk, jj] - ttn[kk2, jj2]
                            tdiv = 3.0
                            #u = mf2 * dnx
                            #a = 9.0
                            #b = 0.0
                            #c = -1.0 * (4.0 * ttn[kk, jj] - ttn[kk2, jj2] + u * slown) ** 2
                            #tref = 0.0
                            #tdiv = 1.0
                        elif nsts[kk, jj] == 0:
                            swsol = 1
                            u = mf2 * dnx
                            a = 1.0
                            b = 0.0
                            c = -slown ** 2 * u ** 2
                            tref = ttn[kk, jj]
                            tdiv = 1.0
                            #u = mf2 * dnx
                            #a = 1.0
                            #b = 0.0
                            #c = -(ttn[kk, jj] + slown * u) ** 2
                            #tref = 0.0
                            #tdiv = 1.0

                    # Now find the solution of the quadratic equation
                    if swsol == 1:
                        rd1 = b ** 2 - 4.0 * a * c
                        if rd1 > 0:
                            tdsh = (-b + math.sqrt(rd1)) / (2.0 * a)
                            trav = (tref + tdsh) / tdiv
                            if tsw2 == 1:
                                travmd = min(trav, travmd)
                            else:
                                travmd = trav
                                tsw2 = 1

    if travmd != 0:
        travmd = min(travm, travmd)
    else:
        travmd = travm

    tsw3 = 0
    travmt = 0
    # wave_ang = round(atand(0.5)) #% KT 31 / 5 / 17 calculate ray angle between neighbouring alive
    # point and node for determination
    wave_ang = round(math.degrees(math.atan(0.5)))
    # eff_ang using fact that arctan(0.5) + arctan(2) = 90 so arctan(2) = -arctan(0.5) % 90
    eff_ang = (-wave_ang - veln[iz, ix]) % 180  # !KT 31 / 15 / 17
    # effective angle by subtracting cell orientation from ray angle
    if velpn[iz, ix] != 0 or stif_den == None:
        angle1 = math.floor(eff_ang)
        angle2 = (angle1 + 1) % 180
        remainder = eff_ang - angle1
        velocity = vel_map[iz, ix] * ((1 - remainder) * avlist2[angle1, velpn[iz, ix]] + remainder * avlist2[angle2, velpn[iz, ix]])
    else:
        # Solves christoffel equation to find group velocity.
        sigma = stif_den[iz, ix, 4]
        if eff_ang % 90 < 0.01 or eff_ang % 90 > 90 - 0.01:
            if abs((eff_ang % 180) - 90) < 1:
                lambda_val = stif_den[iz, ix, 2]
            else:
                lambda_val = stif_den[iz, ix, 0]
            velocity = 1000 * vel_map[iz, ix] * math.sqrt(lambda_val / sigma)
        else:
            c_22 = stif_den[iz, ix, 0]
            c_23 = stif_den[iz, ix, 1]
            c_33 = stif_den[iz, ix, 2]
            c_44 = stif_den[iz, ix, 3]
            tan_ang = math.tan(math.radians(eff_ang))
            A = c_22 + c_33 - 2 * c_44
            B = (c_23 + c_44) * (tan_ang - 1 / tan_ang)
            C = c_22 - c_33
            if eff_ang < 90:
                phase_angle_rad = math.atan((-B - math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
            else:
                phase_angle_rad = math.atan((-B + math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
            lambda_val = 0.5 * (math.cos(2 * phase_angle_rad) * (c_22 - c_44) + math.sin(2 * phase_angle_rad) * (c_23 + c_44) * tan_ang + c_22 + c_44)
            velocity = 1000 * vel_map[iz, ix] * math.sqrt(lambda_val / sigma) / math.cos(math.radians(eff_ang) - phase_angle_rad)

    slown = 1.0 / velocity

    mf2 = math.sqrt(5)  # distance between two diagonally connected points

    j_vec = [ix - 1, ix + 2, ix + 1, ix - 2, ix - 1]
    k_vec = [iz - 2, iz - 1, iz + 2, iz + 1, iz - 2]

    for lp in range(4):  # lp=1:4
        j = j_vec[lp]
        k = k_vec[lp]
        jj = j_vec[lp + 1]
        kk = k_vec[lp + 1]
        if 0 <= j <= nnx - 1:
            if 0 <= k <= nnz - 1:
                if 0 <= jj <= nnx - 1:
                    if 0 <= kk <= nnz - 1:
                        # !There are seven solution options in each quadrant.
                        swsol = 0
                        if nsts[k, j] == 0:
                            swsol = 1
                            if nsts[kk, jj] == 0:
                                u = mf2 * dnx  # !KT 17 / 5 / 17 diagonal nodes are sqrt(2) * dnx from
                                # active node (isotropic grid spacing only)
                                v = mf2 * dnz
                                a = 2
                                b = -2 * (ttn[kk, jj] + ttn[k, j])
                                c = ttn[kk, jj] ** 2 + ttn[k, j] ** 2 - 2 * (u * slown) ** 2
                                # em = ttn(kk, jj) - ttn(k, j);
                                # a = v ** 2 + u ** 2;
                                # b = -2.0 * u ** 2 * em;
                                # c = u ** 2 * (em ** 2 - v ** 2 * slown ** 2);
                                tref = ttn[k, j]
                                tref = 0.0
                            else:
                                u = mf2 * dnx
                                # v = u;
                                a = 1
                                b = 0
                                c = -(slown * u) ** 2
                                tref = ttn[k, j]
                        elif nsts[kk, jj] == 0:
                            swsol = 1
                            u = mf2 * dnx  # !KT 30 / 1 / 17
                            # v = sqrt(2.0) * dnz
                            a = 1
                            b = 0
                            c = -(slown * u) ** 2
                            tref = ttn[kk, jj]
                        # Now find the solution of the quadratic equation
                        if swsol == 1:
                            rd1 = b ** 2 - 4 * a * c
                            if rd1 < 0:
                                rd1 = 0
                            tdsh = (-b + math.sqrt(rd1)) / (2.0 * a)
                            trav = tref + tdsh
                            if tsw3 == 1:
                                travmt = min(trav, travmt)
                            else:
                                travmt = trav
                                tsw3 = 1
    if travmt != 0:
        travmt = min(travmt, travmd)
    else:
        travmt = travmd

    tsw4 = 0
    travms = 0
    # eff_ang = round(mod(wave_ang - veln(iz, ix), 90)) # !KT 31 / 15 / 17 effective angle by
    # subtracting cell orientation from ray angle
    eff_ang = (wave_ang - veln[iz, ix]) % 180
    if velpn[iz, ix] != 0 or stif_den == None:
        angle1 = math.floor(eff_ang)
        angle2 = (angle1 + 1) % 180
        remainder = eff_ang - angle1
        velocity = vel_map[iz, ix] * ((1 - remainder) * avlist2[angle1, velpn[iz, ix]] + remainder * avlist2[angle2, velpn[iz, ix]])
    else:
        # Solves christoffel equation to find group velocity.
        sigma = stif_den[iz, ix, 4]
        if eff_ang % 90 < 0.01 or eff_ang % 90 > 90 - 0.01:
            if abs((eff_ang % 180) - 90) < 1:
                lambda_val = stif_den[iz, ix, 2]
            else:
                lambda_val = stif_den[iz, ix, 0]
            velocity = 1000 * vel_map[iz, ix] * math.sqrt(lambda_val / sigma)
        else:
            c_22 = stif_den[iz, ix, 0]
            c_23 = stif_den[iz, ix, 1]
            c_33 = stif_den[iz, ix, 2]
            c_44 = stif_den[iz, ix, 3]
            tan_ang = math.tan(math.radians(eff_ang))
            A = c_22 + c_33 - 2 * c_44
            B = (c_23 + c_44) * (tan_ang - 1 / tan_ang)
            C = c_22 - c_33
            if eff_ang < 90:
                phase_angle_rad = math.atan((-B - math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
            else:
                phase_angle_rad = math.atan((-B + math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
            lambda_val = 0.5 * (math.cos(2 * phase_angle_rad) * (c_22 - c_44) + math.sin(2 * phase_angle_rad) * (c_23 + c_44) * tan_ang + c_22 + c_44)
            velocity = 1000 * vel_map[iz, ix] * math.sqrt(lambda_val / sigma) / math.cos(math.radians(eff_ang) - phase_angle_rad)

    slown = 1.0 / velocity
    mf2 = math.sqrt(5)  # distance between two diagonally connected points

    j_vec = [ix + 1, ix + 2, ix - 1, ix - 2, ix + 1]
    k_vec = [iz - 2, iz + 1, iz + 2, iz - 1, iz - 2]

    for lp in range(4):  # lp=1:4
        j = j_vec[lp]
        k = k_vec[lp]
        jj = j_vec[lp + 1]
        kk = k_vec[lp + 1]
        if 0 <= j <= nnx - 1:
            if 0 <= k <= nnz - 1:
                if 0 <= jj <= nnx - 1:
                    if 0 <= kk <= nnz - 1:
                        # !There are seven solution options in each quadrant.
                        swsol = 0
                        if nsts[k, j] == 0:
                            swsol = 1
                            if nsts[kk, jj] == 0:
                                u = mf2 * dnx  # !KT 17 / 5 / 17 diagonal nodes are sqrt(2) * dnx from
                                # active node (isotropic grid spacing only)
                                v = mf2 * dnz
                                a = 2
                                b = -2 * (ttn[kk, jj] + ttn[k, j])
                                c = ttn[kk, jj] ** 2 + ttn[k, j] ** 2 - 2 * (u * slown) ** 2
                                # em = ttn[kk, jj] - ttn[k, j]
                                # a = v ** 2 + u ** 2
                                # b = -2.0 * u ** 2 * em
                                # c = u ** 2 * (em ** 2 - v ** 2 * slown ** 2)
                                tref = 0.0
                            else:
                                u = mf2 * dnx
                                # v = u;
                                a = 1
                                b = 0
                                c = -(slown * u) ** 2
                                tref = ttn[k, j]
                        elif nsts[kk, jj] == 0:
                            swsol = 1
                            u = mf2 * dnx  # !KT 30 / 1 / 17
                            # v = sqrt(2.0) * dnz;
                            a = 1
                            b = 0
                            c = -(slown * u) ** 2
                            tref = ttn[kk, jj]
                        # !Now find the solution of the quadratic equation
                        if swsol == 1:
                            rd1 = b ** 2 - 4 * a * c
                            if rd1 < 0:
                                rd1 = 0
                            tdsh = (-b + math.sqrt(rd1)) / (2.0 * a)
                            trav = tref + tdsh
                            if tsw4 == 1:
                                travms = min(trav, travms)
                            else:
                                travms = trav
                                tsw4 = 1
    if travms != 0:
        travms = min(travmt, travms)
    else:
        travms = travmt
    if ttn[iz, ix] != 0:
        travms = min(travms, ttn[iz, ix])
    #ttn[iz, ix] = travms
    return travms


@njit(cache=True)
def update(veln, velpn, vel_map, nsts, ttn, iz, ix, dnx, nnz, nnx, phase_vel, stif_den):
    """
    Our finite difference method for calculating first arrival travel time at point with indices iz, ix from travel times at surrounding points with travel time estimates (not all points have an associated travel time).

    :param veln: Anisotropic orientation of all grid points.
    :type veln: 2D numpy array
    :param velpn: Material index of all grid points (0 if using stiffness tensors and density, otherwise index for column in avlist2).
    :type velpn: 2D numpy array of type int
    :param vel_map: Value used for scaling velocities at all grid points (mainly used for isotropic materials).
    :type vel_map: 2D numpy array
    :param nsts: Node status for points in the array. -1 is for unknown point, if point is still in the heap then value is the position in the tree.
    :type nsts: 2D numpy array of type int
    :param ttn: Current travel time at all points in the grid (0 for far points).
    :type ttn: 2D numpy array
    :param iz: z index of point where finite difference is applied.
    :type iz: int
    :param ix: x index of point where finite difference is applied.
    :type ix: int
    :param dnx: Distance between points in the grid in the x/z direction.
    :type dnx: float
    :param nnz: Number of points in the grid in the z direction.
    :type nnz: int
    :param nnx: Number of points in the grid in the x direction.
    :type nnx: int
    :param phase_vel: Phase velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type phase_vel: 2D numpy array
    :param stif_den: Material parameters with first two indices being the i,j coordinates and the third being the index of the material parameter(c_22, c_23, c_33, c_44, density). Stiffness tensors must be in MPa to avoid overflow errors.
    :type stif_den: 3D numpy array of type np.int64
    :return: Travel time estimate at the point iz, ix.
    :rtype: float
    """
    # First we check which square stencils can be used.
    sten_points = np.zeros(8, dtype=numba.uint8)  # Array for soring how many points required for the stencil have travel time estimates
    #sten_points = np.zeros(8, dtype=int)
    # We find the number of points in each stencil with can be used in the finite difference method.
    if ix > 1:
        if nsts[iz, ix - 2] >= 0:
            sten_points[3] += 1
    if ix > 0:
        if nsts[iz, ix - 1] >= 0:
            sten_points[4] += 1
            sten_points[7] += 1
        if iz > 0:
            if nsts[iz - 1, ix - 1] >= 0:
                sten_points[0] += 1
                sten_points[3] += 1
                sten_points[4] += 1
        if iz < nnz - 1:
            if nsts[iz + 1, ix - 1] >= 0:
                sten_points[2] += 1
                sten_points[3] += 1
                sten_points[7] += 1
    if ix < nnx - 2:
        if nsts[iz, ix + 2] >= 0:
            sten_points[1] += 1
    if ix < nnx - 1:
        if nsts[iz, ix + 1] >= 0:
            sten_points[5] += 1
            sten_points[6] += 1
        if iz > 0:
            if nsts[iz - 1, ix + 1] >= 0:
                sten_points[0] += 1
                sten_points[1] += 1
                sten_points[5] += 1
        if iz < nnz - 1:
            if nsts[iz + 1, ix + 1] >= 0:
                sten_points[1] += 1
                sten_points[2] += 1
                sten_points[6] += 1
    if iz > 1:
        if nsts[iz - 2, ix] >= 0:
            sten_points[0] += 1
    if iz > 0:
        if nsts[iz - 1, ix] >= 0:
            sten_points[4] += 1
            sten_points[5] += 1
    if iz < nnz - 2:
        if nsts[iz + 2, ix] >= 0:
            sten_points[2] += 1
    if iz < nnz - 1:
        if nsts[iz + 1, ix] >= 0:
            sten_points[6] += 1
            sten_points[7] += 1
    # These parameters keep track of the minimum difference and which stencil had the minimum. If they are unchanged then there are no suitable square stencils.
    stencil_no = -1
    min_diff = 1000000.0

    # We go through all square stencils and check if it is possible to use it(all points have travel time estimates) and
    # find the difference in travel time between two points. The stencil with the minimum is used for the finite difference.
    if sten_points[0] == 3:
        diff = abs(ttn[iz - 1, ix - 1] - ttn[iz - 1, ix + 1])
        if diff < min_diff:
            stencil_no = 0
            min_diff = diff
    if sten_points[1] == 3:
        diff = abs(ttn[iz - 1, ix + 1] - ttn[iz + 1, ix + 1])
        if diff < min_diff:
            stencil_no = 1
            min_diff = diff
    if sten_points[2] == 3:
        diff = abs(ttn[iz + 1, ix - 1] - ttn[iz + 1, ix + 1])
        if diff < min_diff:
            stencil_no = 2
            min_diff = diff
    if sten_points[3] == 3:
        diff = abs(ttn[iz - 1, ix - 1] - ttn[iz + 1, ix - 1])
        if diff < min_diff:
            stencil_no = 3
            min_diff = diff
    if sten_points[4] == 3:
        diff = abs(ttn[iz, ix - 1] - ttn[iz - 1, ix])
        if diff < min_diff:
            stencil_no = 4
            min_diff = diff
    if sten_points[5] == 3:
        diff = abs(ttn[iz - 1, ix] - ttn[iz, ix + 1])
        if diff < min_diff:
            stencil_no = 5
            min_diff = diff
    if sten_points[6] == 3:
        diff = abs(ttn[iz + 1, ix] - ttn[iz, ix + 1])
        if diff < min_diff:
            stencil_no = 6
            min_diff = diff
    if sten_points[7] == 3:
        diff = abs(ttn[iz, ix - 1] - ttn[iz + 1, ix])
        if diff < min_diff:
            stencil_no = 7
            min_diff = diff
    #print("Stencil_points :", sten_points)
    #print("Sten no :", stencil_no)
    angle = 0.0
    dist = -1.0
    wavefront_time = 0.0
    if stencil_no != -1:  # If we have a valid stencil we find which stencil in a stencil pair we can use (all stencils are in pairs where only one can be used and use the same points and same value for the difference in the previous section of code) and calculate the angle of the wavefront and the distance to the point we are calculating.
        if stencil_no == 0:
            if ttn[iz - 1, ix - 1] < ttn[iz - 1, ix + 1]:
                if nsts[iz - 1, ix] >= 0:
                    angle, dist = wavefront_angle_dist(ix, iz, ix, ix - 1, ix + 1, iz - 2, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix - 1], ttn[iz - 1, ix + 1])
                    wavefront_time = ttn[iz - 1, ix - 1]
                    #angle, dist = wavefront_angle_dist_diamond(ix, iz, ix, ix - 1, ix + 1, ix, iz - 2, iz - 1, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix - 1], ttn[iz - 1, ix + 1])
                    #wavefront_time = ttn[iz - 1, ix]
                else:
                    angle, dist = wavefront_angle_dist(ix, iz, ix, ix - 1, ix + 1, iz - 2, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix - 1], ttn[iz - 1, ix + 1])
                    wavefront_time = ttn[iz - 1, ix - 1]
            else:
                if nsts[iz - 1, ix] >= 0:
                    angle, dist = wavefront_angle_dist(ix, iz, ix, ix + 1, ix - 1, iz - 2, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix + 1], ttn[iz - 1, ix - 1])
                    wavefront_time = ttn[iz - 1, ix + 1]
                    #angle, dist = wavefront_angle_dist_diamond(ix, iz, ix, ix + 1, ix - 1, ix, iz - 2, iz - 1, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix + 1], ttn[iz - 1, ix - 1])
                    #wavefront_time = ttn[iz - 1, ix]
                else:
                    angle, dist = wavefront_angle_dist(ix, iz, ix, ix + 1, ix - 1, iz - 2, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix + 1], ttn[iz - 1, ix - 1])
                    wavefront_time = ttn[iz - 1, ix + 1]
        elif stencil_no == 1:
            if ttn[iz - 1, ix + 1] < ttn[iz + 1, ix + 1]:
                if nsts[iz, ix + 1] >= 0:
                    angle, dist = wavefront_angle_dist(ix, iz, ix + 2, ix + 1, ix + 1, iz, iz - 1, iz + 1, ttn[iz, ix + 2], ttn[iz - 1, ix + 1], ttn[iz + 1, ix + 1])
                    wavefront_time = ttn[iz - 1, ix + 1]
                    #angle, dist = wavefront_angle_dist_diamond(ix, iz, ix + 2, ix + 1, ix + 1, ix + 1, iz, iz - 1, iz + 1, iz, ttn[iz, ix + 2], ttn[iz - 1, ix + 1], ttn[iz + 1, ix + 1])
                    #wavefront_time = ttn[iz, ix + 1]
                else:
                    angle, dist = wavefront_angle_dist(ix, iz, ix + 2, ix + 1, ix + 1, iz, iz - 1, iz + 1, ttn[iz, ix + 2], ttn[iz - 1, ix + 1], ttn[iz + 1, ix + 1])
                    wavefront_time = ttn[iz - 1, ix + 1]
            else:
                if nsts[iz, ix + 1] >= 0:
                    angle, dist = wavefront_angle_dist(ix, iz, ix + 2, ix + 1, ix + 1, iz, iz + 1, iz - 1, ttn[iz, ix + 2], ttn[iz + 1, ix + 1], ttn[iz - 1, ix + 1])
                    wavefront_time = ttn[iz + 1, ix + 1]
                    #angle, dist = wavefront_angle_dist_diamond(ix, iz, ix + 2, ix + 1, ix + 1, ix + 1, iz, iz + 1, iz - 1, iz, ttn[iz, ix + 2], ttn[iz + 1, ix + 1], ttn[iz - 1, ix + 1])
                    #wavefront_time = ttn[iz, ix + 1]
                else:
                    angle, dist = wavefront_angle_dist(ix, iz, ix + 2, ix + 1, ix + 1, iz, iz + 1, iz - 1, ttn[iz, ix + 2], ttn[iz + 1, ix + 1], ttn[iz - 1, ix + 1])
                    wavefront_time = ttn[iz + 1, ix + 1]
        elif stencil_no == 2:
            if ttn[iz + 1, ix - 1] < ttn[iz + 1, ix + 1]:
                if nsts[iz + 1, ix] >= 0:
                    angle, dist = wavefront_angle_dist(ix, iz, ix, ix - 1, ix + 1, iz + 2, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix - 1], ttn[iz + 1, ix + 1])
                    wavefront_time = ttn[iz + 1, ix - 1]
                    #angle, dist = wavefront_angle_dist_diamond(ix, iz, ix, ix - 1, ix + 1, ix, iz + 2, iz + 1, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix - 1], ttn[iz + 1, ix + 1])
                    #wavefront_time = ttn[iz + 1, ix]
                else:
                    angle, dist = wavefront_angle_dist(ix, iz, ix, ix - 1, ix + 1, iz + 2, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix - 1], ttn[iz + 1, ix + 1])
                    wavefront_time = ttn[iz + 1, ix - 1]
            else:
                if nsts[iz + 1, ix] >= 0:
                    angle, dist = wavefront_angle_dist(ix, iz, ix, ix + 1, ix - 1, iz + 2, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix + 1], ttn[iz + 1, ix - 1])
                    wavefront_time = ttn[iz + 1, ix + 1]
                    #angle, dist = wavefront_angle_dist_diamond(ix, iz, ix, ix + 1, ix - 1, ix, iz + 2, iz + 1, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix + 1], ttn[iz + 1, ix - 1])
                    #wavefront_time = ttn[iz + 1, ix]
                else:
                    angle, dist = wavefront_angle_dist(ix, iz, ix, ix + 1, ix - 1, iz + 2, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix + 1], ttn[iz + 1, ix - 1])
                    wavefront_time = ttn[iz + 1, ix + 1]
        elif stencil_no == 3:
            if ttn[iz - 1, ix - 1] < ttn[iz + 1, ix - 1]:
                if nsts[iz, ix - 1] >= 0:
                    angle, dist = wavefront_angle_dist(ix, iz, ix - 2, ix - 1, ix - 1, iz, iz - 1, iz + 1, ttn[iz, ix - 2], ttn[iz - 1, ix - 1], ttn[iz + 1, ix - 1])
                    wavefront_time = ttn[iz - 1, ix - 1]
                    #angle, dist = wavefront_angle_dist_diamond(ix, iz, ix - 2, ix - 1, ix - 1, ix - 1, iz, iz - 1, iz + 1, iz, ttn[iz, ix - 2], ttn[iz - 1, ix - 1], ttn[iz + 1, ix - 1])
                    #wavefront_time = ttn[iz, ix - 1]
                else:
                    angle, dist = wavefront_angle_dist(ix, iz, ix - 2, ix - 1, ix - 1, iz, iz - 1, iz + 1, ttn[iz, ix - 2], ttn[iz - 1, ix - 1], ttn[iz + 1, ix - 1])
                    wavefront_time = ttn[iz - 1, ix - 1]
            else:
                if nsts[iz, ix - 1] >= 0:
                    angle, dist = wavefront_angle_dist(ix, iz, ix - 2, ix - 1, ix - 1, iz, iz + 1, iz - 1, ttn[iz, ix - 2], ttn[iz + 1, ix - 1], ttn[iz - 1, ix - 1])
                    wavefront_time = ttn[iz + 1, ix - 1]
                    #angle, dist = wavefront_angle_dist_diamond(ix, iz, ix - 2, ix - 1, ix - 1, ix - 1, iz, iz + 1, iz - 1, iz, ttn[iz, ix - 2], ttn[iz + 1, ix - 1], ttn[iz - 1, ix - 1])
                    #wavefront_time = ttn[iz, ix - 1]
                else:
                    angle, dist = wavefront_angle_dist(ix, iz, ix - 2, ix - 1, ix - 1, iz, iz + 1, iz - 1, ttn[iz, ix - 2], ttn[iz + 1, ix - 1], ttn[iz - 1, ix - 1])
                    wavefront_time = ttn[iz + 1, ix - 1]
        elif stencil_no == 4:
            if ttn[iz, ix - 1] < ttn[iz - 1, ix]:
                angle, dist = wavefront_angle_dist(ix, iz, ix - 1, ix - 1, ix, iz - 1, iz, iz - 1, ttn[iz - 1, ix - 1], ttn[iz, ix - 1], ttn[iz - 1, ix])
                wavefront_time = ttn[iz, ix - 1]
            else:
                angle, dist = wavefront_angle_dist(ix, iz, ix - 1, ix, ix - 1, iz - 1, iz - 1, iz, ttn[iz - 1, ix - 1], ttn[iz - 1, ix], ttn[iz, ix - 1])
                wavefront_time = ttn[iz - 1, ix]
        elif stencil_no == 5:
            if ttn[iz - 1, ix] < ttn[iz, ix + 1]:
                angle, dist = wavefront_angle_dist(ix, iz, ix + 1, ix, ix + 1, iz - 1, iz - 1, iz, ttn[iz - 1, ix + 1], ttn[iz - 1, ix], ttn[iz, ix + 1])
                wavefront_time = ttn[iz - 1, ix]
            else:
                angle, dist = wavefront_angle_dist(ix, iz, ix + 1, ix + 1, ix, iz - 1, iz, iz - 1, ttn[iz - 1, ix + 1], ttn[iz, ix + 1], ttn[iz - 1, ix])
                wavefront_time = ttn[iz, ix + 1]
        elif stencil_no == 6:
            if ttn[iz + 1, ix] < ttn[iz, ix + 1]:
                angle, dist = wavefront_angle_dist(ix, iz, ix + 1, ix, ix + 1, iz + 1, iz + 1, iz, ttn[iz + 1, ix + 1], ttn[iz + 1, ix], ttn[iz, ix + 1])
                wavefront_time = ttn[iz + 1, ix]
            else:
                angle, dist = wavefront_angle_dist(ix, iz, ix + 1, ix + 1, ix, iz + 1, iz, iz + 1, ttn[iz + 1, ix + 1], ttn[iz, ix + 1], ttn[iz + 1, ix])
                wavefront_time = ttn[iz, ix + 1]
        elif stencil_no == 7:
            if ttn[iz, ix - 1] < ttn[iz + 1, ix]:
                angle, dist = wavefront_angle_dist(ix, iz, ix - 1, ix - 1, ix, iz + 1, iz, iz + 1, ttn[iz + 1, ix - 1], ttn[iz, ix - 1], ttn[iz + 1, ix])
                wavefront_time = ttn[iz, ix - 1]
            else:
                angle, dist = wavefront_angle_dist(ix, iz, ix - 1, ix, ix - 1, iz + 1, iz + 1, iz, ttn[iz + 1, ix - 1], ttn[iz + 1, ix], ttn[iz, ix - 1])
                wavefront_time = ttn[iz + 1, ix]

    # If no stencils were valid or the grid point is at the edge of the grid we check the triangular stencils.
    if stencil_no == -1 or ix == 0 or ix == nnx - 1 or iz == 0 or iz == nnz - 1: #else:
        #print("try stencils 8-15")

        # We find number of points in the stencils which can be used.
        sten_points = np.zeros(8, dtype=numba.uint8)
        #sten_points = np.zeros(8, dtype=int)
        if ix > 1:
            if nsts[iz, ix - 2] >= 0:
                sten_points[4] += 1
                sten_points[7] += 1
        if ix > 0:
            if nsts[iz, ix - 1] >= 0:
                sten_points[4] += 1
                sten_points[7] += 1
            if iz > 0:
                if nsts[iz - 1, ix - 1] >= 0:
                    sten_points[2] += 1
                    sten_points[7] += 1
            if iz < nnz - 1:
                if nsts[iz + 1, ix - 1] >= 0:
                    sten_points[3] += 1
                    sten_points[4] += 1
        if ix < nnx - 2:
            if nsts[iz, ix + 2] >= 0:
                sten_points[5] += 1
                sten_points[6] += 1
        if ix < nnx - 1:
            if nsts[iz, ix + 1] >= 0:
                sten_points[5] += 1
                sten_points[6] += 1
            if iz > 0:
                if nsts[iz - 1, ix + 1] >= 0:
                    sten_points[1] += 1
                    sten_points[6] += 1
            if iz < nnz - 1:
                if nsts[iz + 1, ix + 1] >= 0:
                    sten_points[0] += 1
                    sten_points[5] += 1
        if iz > 1:
            if nsts[iz - 2, ix] >= 0:
                sten_points[1] += 1
                sten_points[2] += 1
        if iz > 0:
            if nsts[iz - 1, ix] >= 0:
                sten_points[1] += 1
                sten_points[2] += 1
        if iz < nnz - 2:
            if nsts[iz + 2, ix] >= 0:
                sten_points[0] += 1
                sten_points[3] += 1
        if iz < nnz - 1:
            if nsts[iz + 1, ix] >= 0:
                sten_points[0] += 1
                sten_points[3] += 1

        # We look at all valid stencils and find how close the travel times are to the case when our estimated wavefront has perpendicular bisector passing through the point where we are finding the estimated travel time.
        if stencil_no == -1:
            min_diff = 1000000.0
        stencil_no = -2
        if sten_points[0] == 3:
            if ttn[iz + 2, ix] < min(ttn[iz + 1, ix], ttn[iz + 1, ix + 1]):
                #diff = ttn[iz + 1, ix] - ttn[iz + 1, ix + 1]
                diff = abs((math.sqrt(2) - 1) * ttn[iz + 2, ix] + (2 - math.sqrt(2)) * ttn[iz + 1, ix] - ttn[iz + 1, ix + 1])
                if diff < min_diff:
                    stencil_no = 0
                    min_diff = diff
        if sten_points[1] == 3:
            if ttn[iz - 2, ix] < min(ttn[iz - 1, ix], ttn[iz - 1, ix + 1]):
                #diff = ttn[iz - 1, ix] - ttn[iz - 1, ix + 1]
                diff = abs((math.sqrt(2) - 1) * ttn[iz - 2, ix] + (2 - math.sqrt(2)) * ttn[iz - 1, ix] - ttn[iz - 1, ix + 1])
                if diff < min_diff:
                    stencil_no = 1
                    min_diff = diff
        if sten_points[2] == 3:
            if ttn[iz - 2, ix] < min(ttn[iz - 1, ix], ttn[iz - 1, ix - 1]):
                #diff = ttn[iz - 1, ix] - ttn[iz - 1, ix - 1]
                diff = abs((math.sqrt(2) - 1) * ttn[iz - 2, ix] + (2 - math.sqrt(2)) * ttn[iz - 1, ix] - ttn[iz - 1, ix - 1])
                if diff < min_diff:
                    stencil_no = 2
                    min_diff = diff
        if sten_points[3] == 3:
            if ttn[iz + 2, ix] < min(ttn[iz + 1, ix], ttn[iz + 1, ix - 1]):
                #diff = ttn[iz + 1, ix] - ttn[iz + 1, ix - 1]
                diff = abs((math.sqrt(2) - 1) * ttn[iz + 2, ix] + (2 - math.sqrt(2)) * ttn[iz + 1, ix] - ttn[iz + 1, ix - 1])
                if diff < min_diff:
                    stencil_no = 3
                    min_diff = diff
        if sten_points[4] == 3:
            if ttn[iz, ix - 2] < min(ttn[iz, ix - 1], ttn[iz + 1, ix - 1]):
                #diff = ttn[iz, ix - 1] - ttn[iz + 1, ix - 1]
                diff = abs((math.sqrt(2) - 1) * ttn[iz, ix - 2] + (2 - math.sqrt(2)) * ttn[iz, ix - 1] - ttn[iz + 1, ix - 1])
                if diff < min_diff:
                    stencil_no = 4
                    min_diff = diff
        if sten_points[5] == 3:
            if ttn[iz, ix + 2] < min(ttn[iz, ix + 1], ttn[iz + 1, ix + 1]):
                #diff = ttn[iz, ix + 1] - ttn[iz + 1, ix + 1]
                diff = abs((math.sqrt(2) - 1) * ttn[iz, ix + 2] + (2 - math.sqrt(2)) * ttn[iz, ix + 1] - ttn[iz + 1, ix + 1])
                if diff < min_diff:
                    stencil_no = 5
                    min_diff = diff
        if sten_points[6] == 3:
            if ttn[iz, ix + 2] < min(ttn[iz, ix + 1], ttn[iz - 1, ix + 1]):
                #diff = ttn[iz, ix + 1] - ttn[iz - 1, ix + 1]
                diff = abs((math.sqrt(2) - 1) * ttn[iz, ix + 2] + (2 - math.sqrt(2)) * ttn[iz, ix + 1] - ttn[iz - 1, ix + 1])
                if diff < min_diff:
                    stencil_no = 6
                    min_diff = diff
        if sten_points[7] == 3:
            if ttn[iz, ix - 2] < min(ttn[iz, ix - 1], ttn[iz - 1, ix - 1]):
                #diff = ttn[iz, ix - 1] - ttn[iz - 1, ix - 1]
                diff = abs((math.sqrt(2) - 1) * ttn[iz, ix - 2] + (2 - math.sqrt(2)) * ttn[iz, ix - 1] - ttn[iz - 1, ix - 1])
                if diff < min_diff:
                    stencil_no = 7
                    min_diff = diff
        #    stencil_no = -1
        if stencil_no != -2: #  If any stencils are valid, we select the valid stencil in the chosen stencil pair and use it to calculate the angle of the estimated wavefront and the minimum distance to the point we are estimating.
            if stencil_no == 0:
                if ttn[iz + 1, ix] < ttn[iz + 1, ix + 1]:
                    if ix == 0:
                        angle = 90.
                        dist = 1.
                    else:
                        #angle, dist = wavefront_angle_dist2(ix, iz, ix, ix, ix + 1, iz + 2, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix], ttn[iz + 1, ix + 1], 1, -(math.sqrt(2) - 1))
                        angle, dist = wavefront_angle_dist(ix, iz, ix, ix, ix + 1, iz + 2, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix], ttn[iz + 1, ix + 1])
                else:
                    #angle, dist = wavefront_angle_dist2(ix, iz, ix, ix + 1, ix, iz + 2, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix + 1], ttn[iz + 1, ix], 1, -(math.sqrt(2) - 1))
                    angle, dist = wavefront_angle_dist(ix, iz, ix, ix + 1, ix, iz + 2, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix + 1], ttn[iz + 1, ix])
                wavefront_time = ttn[iz + 1, ix + 1]
            elif stencil_no == 1:
                if ttn[iz - 1, ix] < ttn[iz - 1, ix + 1]:
                    if ix == 0:
                        angle = 90.
                        dist = 1.
                    else:
                        #angle, dist = wavefront_angle_dist2(ix, iz, ix, ix, ix + 1, iz - 2, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix], ttn[iz - 1, ix + 1], 1, (math.sqrt(2) - 1))
                        angle, dist = wavefront_angle_dist(ix, iz, ix, ix, ix + 1, iz - 2, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix], ttn[iz - 1, ix + 1])
                    wavefront_time = ttn[iz - 1, ix]
                else:
                    #angle, dist = wavefront_angle_dist2(ix, iz, ix, ix + 1, ix, iz - 2, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix + 1], ttn[iz - 1, ix], 1, (math.sqrt(2) - 1))
                    angle, dist = wavefront_angle_dist(ix, iz, ix, ix + 1, ix, iz - 2, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix + 1], ttn[iz - 1, ix])
                    wavefront_time = ttn[iz - 1, ix + 1]
            elif stencil_no == 2:
                if ttn[iz - 1, ix] < ttn[iz - 1, ix - 1]:
                    if ix == nnx - 1:
                        angle = 90.
                        dist = 1.
                    else:
                        #angle, dist = wavefront_angle_dist2(ix, iz, ix, ix, ix - 1, iz - 2, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix], ttn[iz - 1, ix - 1], 1, -(math.sqrt(2) - 1))
                        angle, dist = wavefront_angle_dist(ix, iz, ix, ix, ix - 1, iz - 2, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix], ttn[iz - 1, ix - 1])
                    wavefront_time = ttn[iz - 1, ix]
                else:
                    #angle, dist = wavefront_angle_dist2(ix, iz, ix, ix - 1, ix, iz - 2, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix - 1], ttn[iz - 1, ix], 1, -(math.sqrt(2) - 1))
                    angle, dist = wavefront_angle_dist(ix, iz, ix, ix - 1, ix, iz - 2, iz - 1, iz - 1, ttn[iz - 2, ix], ttn[iz - 1, ix - 1], ttn[iz - 1, ix])
                    wavefront_time = ttn[iz - 1, ix - 1]
            elif stencil_no == 3:
                if ttn[iz + 1, ix] < ttn[iz + 1, ix - 1]:
                    if ix == nnx - 1:
                        angle = 90.
                        dist = 1.
                    else:
                        #angle, dist = wavefront_angle_dist2(ix, iz, ix, ix, ix - 1, iz + 2, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix], ttn[iz + 1, ix - 1], 1, (math.sqrt(2) - 1))
                        angle, dist = wavefront_angle_dist(ix, iz, ix, ix, ix - 1, iz + 2, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix], ttn[iz + 1, ix - 1])
                    wavefront_time = ttn[iz + 1, ix]
                else:
                    #angle, dist = wavefront_angle_dist2(ix, iz, ix, ix - 1, ix, iz + 2, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix - 1], ttn[iz + 1, ix], 1, (math.sqrt(2) - 1))
                    angle, dist = wavefront_angle_dist(ix, iz, ix, ix - 1, ix, iz + 2, iz + 1, iz + 1, ttn[iz + 2, ix], ttn[iz + 1, ix - 1], ttn[iz + 1, ix])
                    wavefront_time = ttn[iz + 1, ix - 1]
            elif stencil_no == 4:
                if ttn[iz, ix - 1] < ttn[iz + 1, ix - 1]:
                    if iz == 0:
                        angle = 0.
                        dist = 1.
                    else:
                        #angle, dist = wavefront_angle_dist2(ix, iz, ix - 2, ix - 1, ix - 1, iz, iz, iz + 1, ttn[iz, ix - 2], ttn[iz, ix - 1], ttn[iz + 1, ix - 1], (math.sqrt(2) - 1), 1)
                        angle, dist = wavefront_angle_dist(ix, iz, ix - 2, ix - 1, ix - 1, iz, iz, iz + 1, ttn[iz, ix - 2], ttn[iz, ix - 1], ttn[iz + 1, ix - 1])
                    wavefront_time = ttn[iz, ix - 1]
                else:
                    #angle, dist = wavefront_angle_dist2(ix, iz, ix - 2, ix - 1, ix - 1, iz, iz + 1, iz, ttn[iz, ix - 2], ttn[iz + 1, ix - 1], ttn[iz, ix - 1], (math.sqrt(2) - 1), 1)
                    angle, dist = wavefront_angle_dist(ix, iz, ix - 2, ix - 1, ix - 1, iz, iz + 1, iz, ttn[iz, ix - 2], ttn[iz + 1, ix - 1], ttn[iz, ix - 1])
                    wavefront_time = ttn[iz + 1, ix - 1]
            elif stencil_no == 5:
                if ttn[iz, ix + 1] < ttn[iz + 1, ix + 1]:
                    if iz == 0:
                        angle = 0.
                        dist = 1.
                    else:
                        #angle, dist = wavefront_angle_dist2(ix, iz, ix + 2, ix + 1, ix + 1, iz, iz, iz + 1, ttn[iz, ix + 2], ttn[iz, ix + 1], ttn[iz + 1, ix + 1], -(math.sqrt(2) - 1), 1)
                        angle, dist = wavefront_angle_dist(ix, iz, ix + 2, ix + 1, ix + 1, iz, iz, iz + 1, ttn[iz, ix + 2], ttn[iz, ix + 1], ttn[iz + 1, ix + 1])
                    wavefront_time = ttn[iz, ix + 1]
                else:
                    #angle, dist = wavefront_angle_dist2(ix, iz, ix + 2, ix + 1, ix + 1, iz, iz + 1, iz, ttn[iz, ix + 2], ttn[iz + 1, ix + 1], ttn[iz, ix + 1], -(math.sqrt(2) - 1), 1)
                    angle, dist = wavefront_angle_dist(ix, iz, ix + 2, ix + 1, ix + 1, iz, iz + 1, iz, ttn[iz, ix + 2], ttn[iz + 1, ix + 1], ttn[iz, ix + 1])
                    wavefront_time = ttn[iz + 1, ix + 1]
            elif stencil_no == 6:
                if ttn[iz, ix + 1] < ttn[iz - 1, ix + 1]:
                    if iz == nnz - 1:
                        angle = 0.
                        dist = 1.
                    else:
                        #angle, dist = wavefront_angle_dist2(ix, iz, ix + 2, ix + 1, ix + 1, iz, iz, iz - 1, ttn[iz, ix + 2], ttn[iz, ix + 1], ttn[iz - 1, ix + 1], (math.sqrt(2) - 1), 1)
                        angle, dist = wavefront_angle_dist(ix, iz, ix + 2, ix + 1, ix + 1, iz, iz, iz - 1, ttn[iz, ix + 2], ttn[iz, ix + 1], ttn[iz - 1, ix + 1])
                    wavefront_time = ttn[iz, ix + 1]
                else:
                    #angle, dist = wavefront_angle_dist2(ix, iz, ix + 2, ix + 1, ix + 1, iz, iz - 1, iz, ttn[iz, ix + 2], ttn[iz - 1, ix + 1], ttn[iz, ix + 1], (math.sqrt(2) - 1), 1)
                    angle, dist = wavefront_angle_dist(ix, iz, ix + 2, ix + 1, ix + 1, iz, iz - 1, iz, ttn[iz, ix + 2], ttn[iz - 1, ix + 1], ttn[iz, ix + 1])
                    wavefront_time = ttn[iz - 1, ix + 1]
            elif stencil_no == 7:
                if ttn[iz, ix - 1] < ttn[iz - 1, ix - 1]:
                    if iz == nnz - 1:
                        angle = 0.
                        dist = 1.
                    else:
                        #angle, dist = wavefront_angle_dist2(ix, iz, ix - 2, ix - 1, ix - 1, iz, iz, iz - 1, ttn[iz, ix - 2], ttn[iz, ix - 1], ttn[iz - 1, ix - 1], -(math.sqrt(2) - 1), 1)
                        angle, dist = wavefront_angle_dist(ix, iz, ix - 2, ix - 1, ix - 1, iz, iz, iz - 1, ttn[iz, ix - 2], ttn[iz, ix - 1], ttn[iz - 1, ix - 1])
                    wavefront_time = ttn[iz, ix - 1]
                else:
                    #angle, dist = wavefront_angle_dist2(ix, iz, ix - 2, ix - 1, ix - 1, iz, iz - 1, iz, ttn[iz, ix - 2], ttn[iz - 1, ix - 1], ttn[iz, ix - 1], -(math.sqrt(2) - 1), 1)
                    angle, dist = wavefront_angle_dist(ix, iz, ix - 2, ix - 1, ix - 1, iz, iz - 1, iz, ttn[iz, ix - 2], ttn[iz - 1, ix - 1], ttn[iz, ix - 1])
                    wavefront_time = ttn[iz - 1, ix - 1]
            stencil_no += 8
    if dist != -1.0:  # If there is a valid stencil.
        eff_angle = (veln[iz, ix] - angle) % 180  # Apply a rotation to the angle to match the orientation of the material at that point (only used for calculate velocities.

        # If we are not using stiffness tensors then we use the table of phase velocities to find the volocity for the given angle
        if velpn[iz, ix] != 0 or stif_den is None:   #if velpn[iz, ix] != 0 or type(stif_den) == type(None):
            angle1 = math.floor(eff_angle)
            angle2 = (angle1 + 1) % 180
            remainder = eff_angle - angle1
            velocity = vel_map[iz, ix] * ((1 - remainder) * phase_vel[angle1, velpn[iz, ix]] + remainder * phase_vel[angle2, velpn[iz, ix]])
        else:
            #if eff_angle % 90 == 0:
            #    if eff_angle == 90:
            #        val_lambda = stif_den[iz, ix, 2]
            #    else:
            #        val_lambda = stif_den[iz, ix, 0]
            #else:
            #    c_22 = stif_den[iz, ix, 0]
            #    c_23 = stif_den[iz, ix, 1]
            #    c_33 = stif_den[iz, ix, 2]
            #    c_44 = stif_den[iz, ix, 3]
            #    sigma = stif_den[iz, ix, 4]
            #    tan_ang = math.tan(math.radians(eff_angle))
            #    A = c_22 + c_33 - 2 * c_44
            #    B = (c_23 + c_44) * (tan_ang - 1 / tan_ang)
            #    C = c_22 - c_33
            #    if eff_angle < 90:
            #        phase_angle_rad = math.atan((-B - math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
            #    else:
            #        phase_angle_rad = math.atan((-B + math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
            #    lambda_val = 0.5 * (math.cos(2 * phase_angle_rad) * (c_22 - c_44) + math.sin(2 * phase_angle_rad) * (c_23 + c_44) * tan_ang + c_22 + c_44)
            #velocity = vel_map[iz, ix] * math.sqrt(lambda_val / sigma) / math.cos(math.radians(eff_angle) - phase_angle_rad)

            # Find the phase velociy from the stiffness tensors
            cos_ang = math.cos(math.radians(eff_angle))
            sin_ang = math.sin(math.radians(eff_angle))
            # We find the values of a, b and c in the quadratic equation we are solving.
            A = cos_ang ** 2 * stif_den[iz, ix, 0] + sin_ang ** 2 * stif_den[iz, ix, 3]
            B = cos_ang * sin_ang * (stif_den[iz, ix, 1] + stif_den[iz, ix, 3])
            C = cos_ang ** 2 * stif_den[iz, ix, 3] + sin_ang ** 2 * stif_den[iz, ix, 2]
            velocity = 1000 * vel_map[iz, ix] * math.sqrt((A + C + math.sqrt((A - C) ** 2 + 4 * B ** 2)) / (2 * stif_den[iz, ix, 4]))
        return wavefront_time + (dist * dnx / velocity)
    else:
        # if there were no valid stencils we return a travel time of -1.0 to show this.
        return -1.0


@njit(cache=True)
def wavefront_angle_dist(ix, iz, x1, x2, x3, z1, z2, z3, y1, y2, y3):
    """
    Calculates the direction of the normal to the wavefront and the minimum distance between the estimated wavefront and the point we are calculating the travel time for. Stencils have points A, B, C where A<=B<C and the estimated wavefront goes through B.

    :param ix: Index of the x position at the point where we are estimating a travel time.
    :type iz: int
    :param iz: Index of the z position at the point where we are estimating a travel time.
    :type iz: int
    :param x1: Index of the x position at the point with label A.
    :type x1: int
    :param x2: Index of the x position at the point with label B.
    :type x2: int
    :param x3: Index of the x position at the point with label C.
    :type x3: int
    :param z1: Index of the z position at the point with label A.
    :type z1: int
    :param z2: Index of the z position at the point with label B.
    :type z2: int
    :param z3: Index of the z position at the point with label C.
    :type z3: int
    :param y1: Travel time estimate at the point with label A.
    :type y1: float
    :param y2: Travel time estimate at the point with label B.
    :type y2: float
    :param y3: Travel time estimate at the point with label C.
    :type y3: float
    :return: angle, dist - The direction of the normal to the estimated wavefront and the minimum distance between the wavefront and the point we are estimating.
    :rtype: float, float
    """
    # If possible we use linear interpolation to find the point between A and C with the same travel time as B by solving (1-a)y1 + a(y3) = y2.
    if y3 != y1:
        a = (y2 - y1) / (y3 - y1)  # (1-a)y1 + a(y3) = y2
    else:
        return 0.0, -1.0
    # Calculate the position of the point on the estimated wavefront which was obtained using linear interpolation.
    xpos = (1 - a) * x1 + a * x3
    zpos = (1 - a) * z1 + a * z3
    diff_x = x2 - xpos
    diff_z = z2 - zpos
    # Find the angle of the normal to the wavefront.
    if diff_x == 0:
        angle = 0.0
    else:
        angle = (math.degrees(math.atan(diff_z/diff_x)) + 90) % 180
    # Calculate the minimum distance between the estimated wavefront and the point we are estimating.
    dist = abs(diff_z * (x2 - ix) - diff_x * (z2 - iz)) / math.sqrt(diff_x ** 2 + diff_z ** 2)
    return angle, dist


@njit(cache=True)
def travel(scx, scz, nsts, btg, ntr, ttn, veln, velpn, vel_map, stif_den, avlist2, phase_vel, gox, goz, dnx, dnz, nnx, nnz):
    """
    Function for calculating a travel time field for a given source.

    :param scx: x position of the source (gets rounded to the nearest grid point in this implementation).
    :type scx: float
    :param scz: z position of the source (gets rounded to the nearest grid point in this implementation).
    :type scz: float
    :param nsts: Node status for points in the array. -1 is for unknown point, if point is still in the heap then value is the position in the tree.
    :type nsts: 2D numpy array of type int
    :param btg: Positions of points in the min heap (start with empty binary tree).
    :type btg: 2D numpy array of type int
    :param ntr: Number of points in the minimum heap.
    :type ntr: int
    :param ttn: Current travel time at all points in the grid. Start with 2D array of zeros
    :type ttn: 2D numpy array
    :param veln: Anisotropic orientation of all grid points. For isotropic material this parameter will not affect anything.
    :type veln: 2D numpy array
    :param velpn: Material index of all grid points (0 if using stiffness tensors and density, otherwise index for column in avlist2).
    :type velpn: 2D numpy array of type int
    :param vel_map: Value used for scaling velocities at all grid points (mainly used for isotropic materials).
    :type vel_map: 2D numpy array
    :param stif_den: Material parameters with first two indices being the i,j coordinates and the third being the index of the material parameter(c_22, c_23, c_33, c_44, density). Stiffness tensors must be in MPa to avoid overflow errors.
    :type stif_den: 3D numpy array of type np.int64
    :param avlist2: Group velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type avlist2: 2D numpy array
    :param phase_vel: Phase velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type phase_vel: 2D numpy array
    :param gox: x position of the point with indices (0, 0).
    :type gox: float
    :param goz: z position of the point with indices (0, 0).
    :type goz: float
    :param dnx: Distance between points in the grid in the x direction.
    :type dnx: float
    :param dnz: Distance between points in the grid in the z direction.
    :type dnz: float
    :param nnx: Number of points in the grid in the x direction.
    :type nnx: int
    :param nnz: Number of points in the grid in the z direction.
    :type nnz: int
    :return: Travel time field for the given source.
    :rtype: 2D numpy array
    """

    # We determine the nearest point on the grid from the source and use that point as the source (using the exact source location makes the initial propogation around the source more complex).
    isx = round((scx - gox) / dnx)  # round((scx-gox)/dnx)+1
    isz = round((scz - goz) / dnz)  # round((scz-goz)/dnz)+1

    # Set up the variables used around the source for the initial propogation.
    size1 = 2  # Size of grid around source for initial propagation
    subgrid_size1 = 27  # Size of subgrid for initial propagation
    side1 = int((subgrid_size1 - 1) / 2)
    left = max(0, isx - size1)
    right = min(nnx - 1, isx + size1)
    bottom = max(0, isz - size1)
    top = min(nnz - 1, isz + size1)
    temp_isx = isx - left
    temp_isz = isz - bottom
    temp_veln = veln[bottom:top+1, left:right+1]
    temp_velpn = velpn[bottom:top + 1, left:right + 1]
    temp_vel_map = vel_map[bottom:top + 1, left:right + 1]


    veln1 = finer_grid_n(temp_veln, subgrid_size1)
    velpn1 = finer_grid_n(temp_velpn, subgrid_size1)
    vel_map1 = finer_grid_n(temp_vel_map, subgrid_size1, numba.float32)
    #if type(stif_den) != type(None):
    if stif_den != None:
        temp_stif_den = stif_den[bottom:top + 1, left:right + 1, :]
        stif_den1 = finer_grid_n_2(temp_stif_den, subgrid_size1)
    else:
        stif_den1 = None
    ttn1 = np.zeros(veln1.shape)
    nsts1 = - np.ones_like(veln1)
    isx_1 = subgrid_size1 * temp_isx
    isz_1 = subgrid_size1 * temp_isz
    dnx1 = dnx / subgrid_size1
    nnz1 = veln1.shape[0]
    nnx1 = veln1.shape[1]
    max_dist1 = subgrid_size1 * size1  # Maximum distance along horiz and vert

    # For points on the fine grid which have there material parameters copied from the source point we calculate travel times by straight rays since this region is homogeneous
    for i in range(- side1, side1 + 1):
        if 0 <= isz_1 + i <= veln1.shape[0] - 1:
            for j in range(- side1, side1 + 1):
                if 0 <= isx_1 + j <= veln1.shape[1] - 1:
                    diff_z = i
                    diff_x = j
                    # We find the direction of the straight ray
                    if diff_x == 0:#diff_z == 0:
                        angle = 90.0
                    else:
                        #angle = math.degrees(math.atan(diff_x / diff_z))
                        angle = math.degrees(math.atan(diff_z / diff_x))
                    eff_angle = (veln[isz, isx] - angle) % 180
                    if velpn[isz, isx] != 0:  # If not using stiffness tensors.
                        angle1 = math.floor(eff_angle)
                        angle2 = (angle1 + 1) % 180
                        remainder = eff_angle - angle1
                        velocity = vel_map[isz, isx] * ((1 - remainder) * avlist2[angle1, velpn[isz, isx]] + remainder * avlist2[angle2, velpn[isz, isx]])
                    else:
                        # Solving christoffel equation to find group velocities
                        sigma = stif_den[isz, isx, 4]
                        if eff_angle % 90 < 0.01 or eff_angle % 90 > 90 - 0.01:
                            if abs((eff_angle % 180) - 90) < 1:
                                lambda_val = stif_den[isz, isx, 2]
                            else:
                                lambda_val = stif_den[isz, isx, 0]
                            velocity = 1000 * vel_map[isz, isx] * math.sqrt(lambda_val / sigma)
                        else:
                            c_22 = stif_den[isz, isx, 0]
                            c_23 = stif_den[isz, isx, 1]
                            c_33 = stif_den[isz, isx, 2]
                            c_44 = stif_den[isz, isx, 3]
                            tan_ang = math.tan(math.radians(eff_angle))
                            A = c_22 + c_33 - 2 * c_44
                            B = (c_23 + c_44) * (tan_ang - 1/tan_ang)
                            C = c_22 - c_33
                            if eff_angle < 90:
                                phase_angle_rad = math.atan((-B - math.sqrt(B ** 2 + A ** 2 - C ** 2))/(C - A)) % math.pi
                            else:
                                phase_angle_rad = math.atan((-B + math.sqrt(B ** 2 + A ** 2 - C ** 2))/(C - A)) % math.pi
                            lambda_val = 0.5 * (math.cos(2 * phase_angle_rad) * (c_22 - c_44) + math.sin(2 * phase_angle_rad) * (c_23 + c_44) * tan_ang + c_22 + c_44)
                            velocity = 1000 * vel_map[isz, isx] * math.sqrt(lambda_val/sigma)/math.cos(math.radians(eff_angle) - phase_angle_rad)
                    length = dnx1 * math.sqrt(diff_z ** 2 + diff_x ** 2)
                    ttn1[isz_1 + i, isx_1 + j] = length / velocity
                    nsts1[isz_1 + i, isx_1 + j] = 0  # We set the node status to 0 so that the travel times are in the known state (will not be changed).

    # We set up the minimum heap structure used for selecting which point to update.
    snb = 0.5
    maxbt = round(snb * veln1.shape[0] * veln1.shape[1])
    btg1 = np.zeros((maxbt, 2), dtype=numba.int32)
    #btg1 = np.zeros((maxbt, 2), dtype=int)
    ntr1 = 0
    sten_no = 0

    # We add all points on the edge of the starting area to the minimum heap so surrounding points can have the finite difference method applied to them. Since node status is 0 these points will not be updated.
    if isz_1 - side1 >= 0:
        for i in range(max(0, isx_1 - side1), min(nnx1 - 1, isx_1 + side1) + 1):
            nsts1, btg1, ntr1 = addtree(isz_1 - side1, i, nsts1, btg1, ntr1, ttn1)
    if isz_1 + side1 <= nnz1 - 1:
        for i in range(max(0, isx_1 - side1), min(nnx1 - 1, isx_1 + side1) + 1):
            nsts1, btg1, ntr1 = addtree(isz_1 + side1, i, nsts1, btg1, ntr1, ttn1)
    if isx_1 - side1 >= 0:
        for i in range(max(0, isz_1 - side1), min(nnz1 - 1, isz_1 + side1) + 1):
            nsts1, btg1, ntr1 = addtree(i, isx_1 - side1, nsts1, btg1, ntr1, ttn1)
    if isx_1 + side1 <= nnx1 - 1:
        for i in range(max(0, isz_1 - side1), min(nnz1 - 1, isz_1 + side1) + 1):
            nsts1, btg1, ntr1 = addtree(i, isx_1 + side1, nsts1, btg1, ntr1, ttn1)

    #plt.imshow(ttn1)
    #plt.title("ttn1")
    #plt.gca().invert_yaxis()
    #plt.show()

    # While loop that will run until the wavefront encounters the edge of the fine grid around the source.
    finished = False
    while ntr1 > 0 and finished == False:
        # ! Set the "close" point with minimum travel time "alive".
        ix = btg1[1, 1]  # [1,?] as position 1 on binary tree not using 0 index (makes calculating parent and child nodes more complex)
        iz = btg1[1, 0]
        nsts1[iz, ix] = 0

        # Update the binary tree by removing the root and sweeping down the tree.
        nsts1, btg1, ntr1, ttn1 = downtree(nsts1, btg1, ntr1, ttn1)

        # Now update or find values of up to four grid points that surround the new "alive" point.
        # Test points that vary in x
        for i in [ix - 1, ix + 1]:  # =ix-1:2:ix+1
            if 0 <= i <= nnx1 - 1:
                if nsts1[iz, i] == -1:  # When a far point is added to the list of "close" points
                    new_TT = update(veln1, velpn1, vel_map1, nsts1, ttn1, iz, i, dnx1, nnz1, nnx1, phase_vel, stif_den1)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts1, ttn1, dnx1, dnx1, nnx1, nnz1, veln1, velpn1, vel_map1, avlist2, stif_den1)
                    ttn1[iz, i] = new_TT

                    # Add new "close" point to the binary tree.
                    nsts1, btg1, ntr1 = addtree(iz, i, nsts1, btg1, ntr1, ttn1)
                    # plt.imshow(ttn)
                    # plt.show()
                elif nsts1[iz, i] > 0:  # Updating a "close" point.
                    new_TT = update(veln1, velpn1, vel_map1, nsts1, ttn1, iz, i, dnx1, nnx1, nnx1, phase_vel, stif_den1)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts1, ttn1, dnx1, dnx1, nnx1, nnz1, veln1, velpn1, vel_map1, avlist2, stif_den1)
                    ttn1[iz, i] = new_TT
                    nsts1, btg1 = updtree(iz, i, nsts1, btg1, ttn1)

            elif abs(isx_1 - i) == max_dist1 + 1:
                finished = True


        # Test points that vary in z
        for i in [iz - 1, iz + 1]:  # i=iz-1:2:iz+1
            if 0 <= i <= nnz1 - 1:
                if nsts1[i, ix] == -1:  # When a far point is added to the list of "close" points
                    new_TT = update(veln1, velpn1, vel_map1, nsts1, ttn1, i, ix, dnx1, nnz1, nnx1, phase_vel, stif_den1)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts1, ttn1, dnx1, dnx1, nnx1, nnz1, veln1, velpn1, vel_map1, avlist2, stif_den1)
                    ttn1[i, ix] = new_TT
                    # plt.imshow(ttn)
                    # plt.show()
                    # ttn = fouds18(i, ix, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, avlist2)
                    nsts1, btg1, ntr1 = addtree(i, ix, nsts1, btg1, ntr1, ttn1)
                elif nsts1[i, ix] > 0:  # Updating a "close" point.
                    new_TT = update(veln1, velpn1, vel_map1, nsts1, ttn1, i, ix, dnx1, nnz1, nnx1, phase_vel, stif_den1)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts1, ttn1, dnx1, dnx1, nnx1, nnz1, veln1, velpn1, vel_map1, avlist2, stif_den1)
                    ttn1[i, ix] = new_TT
                    nsts1, btg1 = updtree(i, ix, nsts1, btg1, ttn1)
            elif abs(isz_1 - i) == max_dist1 + 1:
                finished = True

    #print(isx_1, isz_1)

    #plt.imshow(ttn1)
    #plt.gca().invert_yaxis()
    #plt.title("ttn1")
    #plt.show()


    # We set up the variables used for a slightly coarser grid as we get further from the source.
    size2 = 6  # Size of grid in each direction around source for propagation (total is 2n+1)
    subgrid_size2 = 9  # Size of subgrid (must be subgrid_size / 3)
    side2 = int((subgrid_size2 - 1) / 2)
    left = max(0, isx - size2)
    right = min(nnx - 1, isx + size2)
    bottom = max(0, isz - size2)
    top = min(nnz - 1, isz + size2)
    temp_isx = isx - left
    temp_isz = isz - bottom
    veln2 = finer_grid_n(veln[bottom:top+1, left:right+1], subgrid_size2)
    velpn2 = finer_grid_n(velpn[bottom:top+1, left:right+1], subgrid_size2)
    vel_map2 = finer_grid_n(vel_map[bottom:top + 1, left:right + 1], subgrid_size2, numba.float32)
    #if type(stif_den) != type(None):
    if stif_den != None:
        stif_den2 = finer_grid_n_2(stif_den[bottom:top + 1, left:right + 1], subgrid_size2)
    else:
        stif_den2 = None
    ttn2 = np.zeros(veln2.shape)
    nsts2 = - np.ones_like(veln2)
    isx_2 = subgrid_size2 * temp_isx
    isz_2 = subgrid_size2 * temp_isz
    dnx2 = dnx / subgrid_size2
    nnz2 = veln2.shape[0]
    nnx2 = veln2.shape[1]
    max_dist2 = subgrid_size2 * size2  # Maximum distance along horiz and vert

    # Set up binary tree
    snb = 0.5
    maxbt = round(snb * veln2.shape[0] * veln2.shape[1])
    btg2 = np.zeros((maxbt, 2), dtype=numba.int32)
    #btg2 = np.zeros((maxbt, 2), dtype=int)
    ntr2 = 0

    # Check which points in the coarser are next to points with unknown travel time and adds them to the binary tree so surounding points can be updated.
    for i in range(0, ttn1.shape[0] + 1, 3):
        for j in range(0, ttn1.shape[1] + 1, 3):
            pos_z = int(isz_2 + (i - isz_1) / 3)
            pos_x = int(isx_2 + (j - isx_1) / 3)
            ttn2[pos_z, pos_x] = ttn1[i, j]
            if nsts1[i, j] == 0:
                nsts2[pos_z, pos_x] = 0
                outer_point = False
                if i - 3 >= 0:
                    if nsts1[i - 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if i + 3 <= nnz1 - 1:
                    if nsts1[i + 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j - 3 >= 0:
                    if nsts1[i, j - 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j + 3 <= nnx1 - 1:
                    if nsts1[i, j + 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if outer_point == True:
                    nsts2, btg2, ntr2 = addtree(pos_z, pos_x, nsts2, btg2, ntr2, ttn2)
                #if nsts1[i - 3, j] == -1 or nsts1[i + 3, j] != -1 or nsts1[i, j - 3] == -1 or nsts1[i, j + 3] == -1:
                #    nsts2, btg2, ntr2 = addtree(pos_z, pos_x, nsts2, btg2, ntr2, ttn2)

            if nsts1[i, j] > 0:
                nsts2, btg2, ntr2 = addtree(pos_z, pos_x, nsts2, btg2, ntr2, ttn2)
            #fix_val = True
            #if nsts1[i, j] == 0:
            #    if i != 0:
            #        if nsts1[pos_z - 3, pos_x] == -1:
            #            fix_val = False
            #    if i != nnz1 - 1:
            #        if nsts1[pos_z + 3, pos_x] == -1:
            #            fix_val = False
            #    if j != 0:
            #        if nsts1[pos_z, pos_x - 3] == -1:
            #            fix_val = False
            #    if j != nnx1 - 1:
            #        if nsts1[pos_z, pos_x + 3] == -1:
            #            fix_val = False
            #    if fix_val == True:
            #        nsts2[pos_z, pos_x] = 0
            #elif nsts1[i, j] != -1 or fix_val == False:
            #    nsts2, btg2, ntr2 = addtree(i, j, nsts2, btg2, ntr2, ttn2)

    #plt.figure(figsize=(8, 8))
    #plt.imshow(ttn2)
    #plt.gca().invert_yaxis()
    #plt.title("ttn2")
    #plt.draw()

    #plt.figure(figsize=(8, 8))
    #plt.imshow(nsts2, vmax=1)
    #plt.gca().invert_yaxis()
    #plt.title("nsts2")
    #plt.show()

    # Loop through until boundary encountered.
    finished = False
    while ntr2 > 0 and finished == False:
        # if math.sqrt(points) % 1 == 0:
        #    print(points)
        #    plt.imshow(ttn)
        #    plt.draw()
        #    plt.pause(0.01)
        #    plt.clf()
        # plt.close()

        # We set the "close" point with minimum traveltime to "alive"
        ix = btg2[1, 1]  # [1,?] as position 1 on binary tree not using 0 index (makes calculating parent and child nodes more complex.
        iz = btg2[1, 0]
        nsts2[iz, ix] = 0

        # Update the binary tree by removing the root and sweeping down the tree.
        nsts2, btg2, ntr2, ttn2 = downtree(nsts2, btg2, ntr2, ttn2)

        # Update or find values of up to four grid points that surround the new "alive" point.
        # Test points that vary in x
        for i in [ix - 1, ix + 1]:  # =ix-1:2:ix+1
            if 0 <= i <= nnx2 - 1:
                if nsts2[iz, i] == -1:  # If far point is added to the list of "close" points
                    new_TT = update(veln2, velpn2, vel_map2, nsts2, ttn2, iz, i, dnx2, nnz2, nnx2, phase_vel, stif_den2)
                    if new_TT == -1.0:  # if no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts2, ttn2, dnx2, dnx2, nnx2, nnz2, veln2, velpn2, vel_map2, avlist2, stif_den2)
                    ttn2[iz, i] = new_TT
                    nsts2, btg2, ntr2 = addtree(iz, i, nsts2, btg2, ntr2, ttn2)
                    # plt.imshow(ttn)
                    # plt.show()
                elif nsts2[iz, i] > 0:  # "close" point is updated
                    new_TT = update(veln2, velpn2, vel_map2, nsts2, ttn2, iz, i, dnx2, nnz2, nnx2, phase_vel, stif_den2)
                    if new_TT == -1.0:  # if no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts2, ttn2, dnx2, dnx2, nnx2, nnz2, veln2, velpn2, vel_map2, avlist2, stif_den2)
                    ttn2[iz, i] = new_TT
                    nsts2, btg2 = updtree(iz, i, nsts2, btg2, ttn2)
            elif abs(isx_2 - i) == max_dist2 + 1:
                finished = True


        # Test points that vary in z
        for i in [iz - 1, iz + 1]:  # i=iz-1:2:iz+1
            if 0 <= i <= nnz2 - 1:
                if nsts2[i, ix] == -1:  # Far point is added to the list of "close" points
                    new_TT = update(veln2, velpn2, vel_map2, nsts2, ttn2, i, ix, dnx2, nnz2, nnx2, phase_vel, stif_den2)
                    if new_TT == -1.0:  # if no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts2, ttn2, dnx2, dnx2, nnx2, nnz2, veln2, velpn2, vel_map2, avlist2, stif_den2)
                    ttn2[i, ix] = new_TT
                    # plt.imshow(ttn)
                    # plt.show()
                    nsts2, btg2, ntr2 = addtree(i, ix, nsts2, btg2, ntr2, ttn2)
                elif nsts2[i, ix] > 0:  # "close" point is updated
                    new_TT = update(veln2, velpn2, vel_map2, nsts2, ttn2, i, ix, dnx2, nnz2, nnx2, phase_vel, stif_den2)
                    if new_TT == -1.0:  # if no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts2, ttn2, dnx2, dnx2, nnx2, nnz2, veln2, velpn2, vel_map2, avlist2, stif_den2)
                    ttn2[i, ix] = new_TT
                    nsts2, btg2 = updtree(i, ix, nsts2, btg2, ttn2)
            elif abs(isz_2 - i) == max_dist2 + 1:
                finished = True

    #plt.imshow(ttn2)
    #plt.title("ttn2")
    #plt.gca().invert_yaxis()
    #plt.show()

    # Set up variables required for the last finer grid before moving to the original grid.
    size3 = 13  # Size of grid in each direction around source for propagation (total is 2n+1)
    subgrid_size3 = 3  # Size of subgrid (must be subgrid_size / 3)
    side = int((subgrid_size3 - 1) / 2)
    left = max(0, isx - size3)
    right = min(nnx - 1, isx + size3)
    bottom = max(0, isz - size3)
    top = min(nnz - 1, isz + size3)
    temp_isx = isx - left
    temp_isz = isz - bottom
    veln3 = finer_grid_n(veln[bottom:top+1, left:right+1], subgrid_size3)
    velpn3 = finer_grid_n(velpn[bottom:top+1, left:right+1], subgrid_size3)
    vel_map3 = finer_grid_n(vel_map[bottom:top + 1, left:right + 1], subgrid_size3, numba.float32)
    #if type(stif_den) != type(None):
    if stif_den != None:
        stif_den3 = finer_grid_n_2(stif_den[bottom:top + 1, left:right + 1], subgrid_size3)
    else:
        stif_den3 = None
    ttn3 = np.zeros(veln3.shape)
    nsts3 = - np.ones_like(veln3)
    isx_3 = subgrid_size3 * temp_isx
    isz_3 = subgrid_size3 * temp_isz
    dnx3 = dnx / subgrid_size3
    dnz3 = dnz / subgrid_size3
    nnz3 = veln3.shape[0]
    nnx3 = veln3.shape[1]
    max_dist3 = subgrid_size3 * size3  # Maximum distance along horiz and vert

    snb = 0.5
    maxbt = round(snb * veln3.shape[0] * veln3.shape[1])
    btg3 = np.zeros((maxbt, 2), dtype=numba.int32)
    #btg3 = np.zeros((maxbt, 2), dtype=int)
    ntr3 = 0
    sten_no = 0

    # Find the points with surounding points that need updating and add to binary tree.
    for i in range(0, ttn2.shape[0] + 1, 3):
        for j in range(0, ttn2.shape[1] + 1, 3):
            pos_z = int(isz_3 + (i - isz_2) / 3)
            pos_x = int(isx_3 + (j - isx_2) / 3)
            ttn3[pos_z, pos_x] = ttn2[i, j]
            if nsts2[i, j] == 0:
                nsts3[pos_z, pos_x] = 0
                outer_point = False
                if i - 3 >= 0:
                    if nsts2[i - 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if i + 3 <= nnz2 - 1:
                    if nsts2[i + 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j - 3 >= 0:
                    if nsts2[i, j - 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j + 3 <= nnx2 - 1:
                    if nsts2[i, j + 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if outer_point == True:
                    nsts3, btg3, ntr3 = addtree(pos_z, pos_x, nsts3, btg3, ntr3, ttn3)
                # if nsts1[i - 3, j] == -1 or nsts1[i + 3, j] != -1 or nsts1[i, j - 3] == -1 or nsts1[i, j + 3] == -1:
                #    nsts2, btg2, ntr2 = addtree(pos_z, pos_x, nsts2, btg2, ntr2, ttn2)

            if nsts2[i, j] > 0:
                nsts3, btg3, ntr3 = addtree(pos_z, pos_x, nsts3, btg3, ntr3, ttn3)

    #plt.figure(figsize=(8, 8))
    #plt.imshow(ttn3)
    #plt.gca().invert_yaxis()
    #plt.title("ttn3")
    #plt.draw()

    #plt.figure(figsize=(8, 8))
    #plt.imshow(nsts3, vmax=1)
    #plt.gca().invert_yaxis()
    #plt.title("nsts3")
    #plt.show()

    # Update points until we hit a boundary.
    finished = False
    while ntr3 > 0 and finished == False:
        # if math.sqrt(points) % 1 == 0:
        #    print(points)
        #    plt.imshow(ttn)
        #    plt.draw()
        #    plt.pause(0.01)
        #    plt.clf()
        # plt.close()

        # Set the "close" point with minimum traveltime to "alive".
        ix = btg3[1, 1]  # [1,?] as position 1 on binary tree not using 0 index (makes calculating parent and child nodes more complex
        iz = btg3[1, 0]
        nsts3[iz, ix] = 0
        # Update the binary tree by removing the root and sweeping down the tree.
        nsts3, btg3, ntr3, ttn3 = downtree(nsts3, btg3, ntr3, ttn3)

        # Now update or find values of up to four grid points that surround the new "alive" point.
        # Test points that vary in x
        for i in [ix - 1, ix + 1]:  # =ix-1:2:ix+1
            if 0 <= i <= nnx3 - 1:
                if nsts3[iz, i] == -1:  # far point is added to the list of "close" points
                    new_TT = update(veln3, velpn3, vel_map3, nsts3, ttn3, iz, i, dnx3, nnz3, nnx3, phase_vel, stif_den3)
                    if new_TT == -1.0:  # if no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts3, ttn3, dnx3, dnx3, nnx3, nnz3, veln3, velpn3, vel_map3, avlist2, stif_den3)
                    ttn3[iz, i] = new_TT
                    nsts3, btg3, ntr3 = addtree(iz, i, nsts3, btg3, ntr3, ttn3)
                    # plt.imshow(ttn)
                    # plt.show()
                elif nsts3[iz, i] > 0:  # "close" point is updated
                    new_TT = update(veln3, velpn3, vel_map3, nsts3, ttn3, iz, i, dnx3, nnz3, nnx3, phase_vel, stif_den3)
                    if new_TT == -1.0:  # if no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts3, ttn3, dnx3, dnx3, nnx3, nnz3, veln3, velpn3, vel_map3, avlist2, stif_den3)
                    ttn3[iz, i] = new_TT
                    nsts3, btg3 = updtree(iz, i, nsts3, btg3, ttn3)
            elif abs(isx_3 - i) == max_dist3 + 1:
                finished = True


        # Test points that vary in z.
        for i in [iz - 1, iz + 1]:  # i=iz-1:2:iz+1
            if 0 <= i <= nnz3 - 1:
                if nsts3[i, ix] == -1:  # Far point is added to the list of "close" points
                    new_TT = update(veln3, velpn3, vel_map3, nsts3, ttn3, i, ix, dnx3, nnz3, nnx3, phase_vel, stif_den3)
                    if new_TT == -1.0:  # if no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts3, ttn3, dnx3, dnx3, nnx3, nnz3, veln3, velpn3, vel_map3, avlist2, stif_den3)
                    ttn3[i, ix] = new_TT
                    # plt.imshow(ttn)
                    # plt.show()
                    nsts3, btg3, ntr3 = addtree(i, ix, nsts3, btg3, ntr3, ttn3)
                elif nsts3[i, ix] > 0:  # "close" point is updated
                    new_TT = update(veln3, velpn3, vel_map3, nsts3, ttn3, i, ix, dnx3, nnz3, nnx3, phase_vel, stif_den3)
                    if new_TT == -1.0:  # if no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts3, ttn3, dnx3, dnx3, nnx3, nnz3, veln3, velpn3, vel_map3, avlist2, stif_den3)
                    ttn3[i, ix] = new_TT
                    nsts3, btg3 = updtree(i, ix, nsts3, btg3, ttn3)
            elif abs(isz_3 - i) == max_dist3 + 1:
                finished = True

    #plt.figure(figsize=(8, 8))
    #plt.imshow(ttn3)
    #plt.gca().invert_yaxis()
    #plt.title("ttn3")
    #plt.show()

    #nsts = - np.ones_like(veln)
    nsts = - np.ones(veln.shape, dtype=numba.int32)

    # Move back onto original grid
    # Find points which have surrounding points that need updating and add them to the min heap.
    for i in range(0, ttn3.shape[0] + 1, 3):
        for j in range(0, ttn3.shape[1] + 1, 3):
            pos_z = int(isz + (i - isz_3) / 3)
            pos_x = int(isx + (j - isx_3) / 3)
            ttn[pos_z, pos_x] = ttn3[i, j]
            if nsts3[i, j] == 0:
                nsts[pos_z, pos_x] = 0
                outer_point = False
                if i - 3 >= 0:
                    if nsts3[i - 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if i + 3 <= nnz3 - 1:
                    if nsts3[i + 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j - 3 >= 0:
                    if nsts3[i, j - 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j + 3 <= nnx3 - 1:
                    if nsts3[i, j + 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if outer_point == True:
                    nsts, btg, ntr = addtree(pos_z, pos_x, nsts, btg, ntr, ttn)
                #if nsts1[i - 3, j] == -1 or nsts1[i + 3, j] != -1 or nsts1[i, j - 3] == -1 or nsts1[i, j + 3] == -1:
                #    nsts2, btg2, ntr2 = addtree(pos_z, pos_x, nsts2, btg2, ntr2, ttn2)

            if nsts3[i, j] > 0:
                nsts, btg, ntr = addtree(pos_z, pos_x, nsts, btg, ntr, ttn)

    #plt.figure(figsize=(8, 8))
    #plt.imshow(ttn)
    #plt.gca().invert_yaxis()
    #plt.title("ttn")
    #plt.draw()

    #plt.figure(figsize=(8, 8))
    #plt.imshow(nsts, vmax=1)
    #plt.gca().invert_yaxis()
    #plt.title("nsts")
    #plt.show()

    # Run until binary tree is empty (i.e all points have travel times).
    while ntr > 0:
        # Set the "close" point with minimum traveltime to "alive"
        ix = btg[1, 1]  # [1,?] as position 1 on binary tree not using 0 index (makes calculating parent and child nodes more complex)
        iz = btg[1, 0]
        nsts[iz, ix] = 0
        # Update the binary tree by removing the root and sweeping down the tree.
        nsts, btg, ntr, ttn = downtree(nsts, btg, ntr, ttn)

        # Now update or find values of up to four grid points that surround the new "alive" point.
        # Test points that vary in x
        for i in [ix - 1, ix + 1]:  # =ix-1:2:ix+1
            if 0 <= i <= nnx - 1:
                if nsts[iz, i] == -1:  # far point is added to the list of "close" points
                    new_TT = update(veln, velpn, vel_map, nsts, ttn, iz, i, dnx, nnz, nnx, phase_vel, stif_den)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, vel_map, avlist2, stif_den)
                    ttn[iz, i] = new_TT
                    nsts, btg, ntr = addtree(iz, i, nsts, btg, ntr, ttn)
                    #plt.imshow(ttn)
                    #plt.show()
                elif nsts[iz, i] > 0:
                    # This happens when a "close" point is updated
                    new_TT = update(veln, velpn, vel_map, nsts, ttn, iz, i, dnx, nnz, nnx, phase_vel, stif_den)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, vel_map, avlist2, stif_den)
                        #sten_no = -10
                    ttn[iz, i] = new_TT
                    nsts, btg = updtree(iz, i, nsts, btg, ttn)


        # Test points that vary in z
        for i in [iz - 1, iz + 1]:  # i=iz-1:2:iz+1
            if 0 <= i <= nnz - 1:
                if nsts[i, ix] == -1:  # "far" point is added to the list of "close" points
                    new_TT = update(veln, velpn, vel_map, nsts, ttn, i, ix, dnx, nnz, nnx, phase_vel, stif_den)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, vel_map, avlist2, stif_den)
                    ttn[i, ix] = new_TT
                    #plt.imshow(ttn)
                    #plt.show()
                    #ttn = fouds18(i, ix, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, avlist2)
                    nsts, btg, ntr = addtree(i, ix, nsts, btg, ntr, ttn)
                elif nsts[i, ix] > 0:  # "close" point is updated.
                    new_TT = update(veln, velpn, vel_map, nsts, ttn, i, ix, dnx, nnz, nnx, phase_vel, stif_den)
                    if new_TT == -1.0:
                        new_TT = fouds18_A(i, ix, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, vel_map, avlist2, stif_den)
                    ttn[i, ix] = new_TT
                    nsts, btg = updtree(i, ix, nsts, btg, ttn)


    #plt.figure(figsize=(8, 8))
    #plt.imshow(ttn)
    #plt.gca().invert_yaxis()
    #plt.title("ttn")
    #plt.draw()

    #plt.figure(figsize=(8, 8))
    #plt.contourf(ttn, 30)
    #plt.gca().invert_yaxis()
    #plt.title("ttn")
    #plt.show()

    return ttn


@njit(cache=True)
def travel_finer_grid(scx, scz, veln0, velpn0, vel_map0, stif_den0, subgrid_size, avlist2, phase_vel, gox, goz, dnx, dnz):
    """
    Function for calculating a travel time field for a given source on a finer grid.

    :param scx: x position of the source (gets rounded to the nearest grid point in this implementation).
    :type scx: float
    :param scz: z position of the source (gets rounded to the nearest grid point in this implementation).
    :type scz: float
    :param veln0: Anisotropic orientation of all grid points on the original grid (not finer grid).
    :type veln0: 2D numpy array
    :param velpn0: Material index of all grid points on the original grid (not finer grid). Values are 0 if using stiffness tensors and density, otherwise index for column in avlist2.
    :type velpn0: 2D numpy array of type int
    :param vel_map0: Value used for scaling velocities at all grid points on the original grid (not finer grid). This is mainly used for isotropic materials.
    :type vel_map0: 2D numpy array
    :param stif_den0: Material parameters with first two indices being the i,j coordinates and the third being the index of the material parameter(c_22, c_23, c_33, c_44, density). Stiffness tensors must be in MPa to avoid overflow errors.
    :type stif_den0: 3D numpy array of type int64
    :param subgrid_size: The size increase of the finer grid. Must be an odd integer so that points match in the original and finer grid.
    :type subgrid_size: int
    :param avlist2: Group velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type avlist2: 2D numpy array
    :param phase_vel: Phase velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type phase_vel: 2D numpy array
    :param gox: x position of the point with indices (0, 0).
    :type gox: float
    :param goz: z position of the point with indices (0, 0).
    :type goz: float
    :param dnx: Distance between points in the grid in the x direction.
    :type dnx: float
    :param dnz: Distance between points in the grid in the z direction. Must equal dnx.
    :type dnx: float
    :return: Travel time field for the required source on the finer grid.
    :rtype: 2D numpy array
    """

    # Generate the finer grid and assign material properties.
    veln = finer_grid_n(veln0, subgrid_size)
    velpn = finer_grid_n(velpn0, subgrid_size)
    vel_map = finer_grid_n(vel_map0, subgrid_size, numba.float32)
    if stif_den0 is None:
        stif_den = np.zeros((veln.shape[0], veln.shape[1], 5), dtype=numba.int64)
        #stif_den = np.zeros((veln.shape[0], veln.shape[1], 5), dtype=np.int64)
    else:
        stif_den = finer_grid_n_2(stif_den0, subgrid_size)

    #nnx = veln.shape[1]
    #nnz = veln.shape[0]
    [nnz, nnx] = veln.shape

    # Find the nearest grid point from the source on the original grid and find the corresponding location on he finer grid.
    isx0 = round((scx - gox) / dnx)  # round((scx-gox)/dnx)+1
    isz0 = round((scz - goz) / dnz)  # round((scz-goz)/dnz)+1

    isx = subgrid_size * isx0
    isz = subgrid_size * isz0
    #print(isx0, isz0, veln0.shape)
    #print(isx, isz, nnx, nnz)

    # Set up min heap variables.
    snb = 0.25
    maxbt = round(snb * nnx * nnz)
    ntr = 0
    ttn = np.zeros(veln.shape)
    #btg = np.zeros((maxbt, 2), dtype=int)
    btg = np.zeros((maxbt, 2), dtype=numba.int32)


    # Set up finer grid around the source.
    size1 = 2 * subgrid_size + int((subgrid_size - 1) / 2)  # Size of grid around source for initial propagation
    subgrid_size1 = 9  # Size of subgrid for initial propagation
    side1 = int((subgrid_size1 - 1) / 2) + subgrid_size1 * int((subgrid_size - 1) / 2)
    left = max(0, isx - size1)
    right = min(nnx - 1, isx + size1)
    bottom = max(0, isz - size1)
    top = min(nnz - 1, isz + size1)
    temp_isx = isx - left
    temp_isz = isz - bottom
    temp_veln = veln[bottom:top+1, left:right+1]
    temp_velpn = velpn[bottom:top + 1, left:right + 1]
    temp_vel_map = vel_map[bottom:top + 1, left:right + 1]


    veln1 = finer_grid_n(temp_veln, subgrid_size1)
    velpn1 = finer_grid_n(temp_velpn, subgrid_size1)
    vel_map1 = finer_grid_n(temp_vel_map, subgrid_size1, numba.float32)
    #if type(stif_den) != type(None):
    if stif_den is None:
        stif_den1 = None
    else:
        #print(stif_den)
        #temp_stif_den = stif_den[bottom:top + 1, left:right + 1]
        stif_den1 = finer_grid_n_2(stif_den[bottom:top + 1, left:right + 1], subgrid_size1)
        #stif_den1 = finer_grid_n_2(stif_den, subgrid_size1)
    ttn1 = np.zeros(veln1.shape)
    nsts1 = - np.ones_like(veln1)
    isx_1 = subgrid_size1 * temp_isx
    isz_1 = subgrid_size1 * temp_isz
    dnx1 = dnx / subgrid_size1
    nnz1 = veln1.shape[0]
    nnx1 = veln1.shape[1]
    max_dist1 = subgrid_size1 * size1  # Maximum distance along horiz and vert

    # For points which obtained their material properties from the source point, we find their travel time using a straight ray from the source as this region is homogenious.
    for i in range(- side1, side1 + 1):
        if 0 <= isz_1 + i <= veln1.shape[0] - 1:
            for j in range(- side1, side1 + 1):
                if 0 <= isx_1 + j <= veln1.shape[1] - 1:
                    diff_z = i
                    diff_x = j
                    if diff_x == 0:#diff_z == 0:
                        angle = 90.0
                    else:
                        #angle = math.degrees(math.atan(diff_x / diff_z))
                        angle = math.degrees(math.atan(diff_z / diff_x))
                    eff_angle = (veln[isz, isx] + angle) % 180
                    if velpn[isz, isx] != 0:
                        angle1 = math.floor(eff_angle)
                        angle2 = (angle1 + 1) % 180
                        remainder = eff_angle - angle1
                        velocity = vel_map[isz, isx] * ((1 - remainder) * avlist2[angle1, velpn[isz, isx]] + remainder * avlist2[angle2, velpn[isz, isx]])
                    else:
                        # Solve the christoffel equation for the group velocity.
                        sigma = stif_den[isz, isx, 4]
                        if eff_angle % 90 < 0.01 or eff_angle % 90 > 90 - 0.01:
                            if abs((eff_angle % 180) - 90) < 1:
                                lambda_val = stif_den[isz, isx, 2]
                            else:
                                lambda_val = stif_den[isz, isx, 0]
                            velocity = 1000 * vel_map[isz, isx] * math.sqrt(lambda_val / sigma)
                        else:
                            c_22 = stif_den[isz, isx, 0]
                            c_23 = stif_den[isz, isx, 1]
                            c_33 = stif_den[isz, isx, 2]
                            c_44 = stif_den[isz, isx, 3]
                            tan_ang = math.tan(math.radians(eff_angle))
                            A = c_22 + c_33 - 2 * c_44
                            B = (c_23 + c_44) * (tan_ang - 1/tan_ang)
                            C = c_22 - c_33
                            if eff_angle < 90:
                                phase_angle_rad = math.atan((-B - math.sqrt(B ** 2 + A ** 2 - C ** 2))/(C - A)) % math.pi
                            else:
                                phase_angle_rad = math.atan((-B + math.sqrt(B ** 2 + A ** 2 - C ** 2))/(C - A)) % math.pi
                            lambda_val = 0.5 * (math.cos(2 * phase_angle_rad) * (c_22 - c_44) + math.sin(2 * phase_angle_rad) * (c_23 + c_44) * tan_ang + c_22 + c_44)
                            velocity = 1000 * vel_map[isz, isx] * math.sqrt(lambda_val/sigma)/math.cos(math.radians(eff_angle) - phase_angle_rad)
                    length = dnx1 * math.sqrt(diff_z ** 2 + diff_x ** 2)
                    ttn1[isz_1 + i, isx_1 + j] = length / velocity
                    # Change node status to 0, so travel times will not be updated.
                    nsts1[isz_1 + i, isx_1 + j] = 0

    # Set up min heap for fine grid around the source.
    snb = 0.25
    maxbt = round(snb * veln1.shape[0] * veln1.shape[1])
    #btg1 = np.zeros((maxbt, 2), dtype=int)
    btg1 = np.zeros((maxbt, 2), dtype=numba.int32)
    ntr1 = 0

    # Points with travel times which are next to a point without a travel time are added to the min heap so that surrounding points can obtain travel times.
    if isz_1 - side1 >= 0:
        for i in range(max(0, isx_1 - side1), min(nnx1 - 1, isx_1 + side1) + 1):
            nsts1, btg1, ntr1 = addtree(isz_1 - side1, i, nsts1, btg1, ntr1, ttn1)
    if isz_1 + side1 <= nnz1 - 1:
        for i in range(max(0, isx_1 - side1), min(nnx1 - 1, isx_1 + side1) + 1):
            nsts1, btg1, ntr1 = addtree(isz_1 + side1, i, nsts1, btg1, ntr1, ttn1)
    if isx_1 - side1 >= 0:
        for i in range(max(0, isz_1 - side1), min(nnz1 - 1, isz_1 + side1) + 1):
            nsts1, btg1, ntr1 = addtree(i, isx_1 - side1, nsts1, btg1, ntr1, ttn1)
    if isx_1 + side1 <= nnx1 - 1:
        for i in range(max(0, isz_1 - side1), min(nnz1 - 1, isz_1 + side1) + 1):
            nsts1, btg1, ntr1 = addtree(i, isx_1 + side1, nsts1, btg1, ntr1, ttn1)

    # Update/Find travel times until we reach the boundary of the starting grid.
    finished = False
    while ntr1 > 0 and finished == False:
        # if math.sqrt(points) % 1 == 0:
        #    print(points)
        #    plt.imshow(ttn)
        #    plt.draw()
        #    plt.pause(0.01)
        #    plt.clf()
        # plt.close()

        # Set the "close" point with minimum traveltime to "alive"
        ix = btg1[1, 1]  # [1,?] as position 1 on binary tree not using 0 index (makes calculating parent and child nodes more complex).
        iz = btg1[1, 0]
        nsts1[iz, ix] = 0

        # Update the binary tree by removing the root and sweeping down the tree.
        nsts1, btg1, ntr1, ttn1 = downtree(nsts1, btg1, ntr1, ttn1)

        # Update or find values of up to four grid points that surround the new "alive" point.
        # Test points that vary in x
        for i in [ix - 1, ix + 1]:
            if 0 <= i <= nnx1 - 1:
                if nsts1[iz, i] == -1:  # "far" point is added to the list of "close" points
                    new_TT = update(veln1, velpn1, vel_map1, nsts1, ttn1, iz, i, dnx1, nnz1, nnx1, phase_vel, stif_den1)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts1, ttn1, dnx1, dnx1, nnx1, nnz1, veln1, velpn1, vel_map1, avlist2, stif_den1)
                    ttn1[iz, i] = new_TT
                    nsts1, btg1, ntr1 = addtree(iz, i, nsts1, btg1, ntr1, ttn1)
                elif nsts1[iz, i] > 0:  # "close" point is updated
                    new_TT = update(veln1, velpn1, vel_map1, nsts1, ttn1, iz, i, dnx1, nnz1, nnx1, phase_vel, stif_den1)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts1, ttn1, dnx1, dnx1, nnx1, nnz1, veln1, velpn1, vel_map1, avlist2, stif_den1)
                    ttn1[iz, i] = new_TT
                    nsts1, btg1 = updtree(iz, i, nsts1, btg1, ttn1)

            elif abs(isx_1 - i) == max_dist1 + 1:
                finished = True


        # Test points that vary in z
        for i in [iz - 1, iz + 1]:
            if 0 <= i <= nnz1 - 1:
                if nsts1[i, ix] == -1:  # "far" point is added to the list of "close" points
                    new_TT = update(veln1, velpn1, vel_map1, nsts1, ttn1, i, ix, dnx1, nnz1, nnx1, phase_vel, stif_den1)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts1, ttn1, dnx1, dnx1, nnx1, nnz1, veln1, velpn1, vel_map1, avlist2, stif_den1)
                    ttn1[i, ix] = new_TT
                    nsts1, btg1, ntr1 = addtree(i, ix, nsts1, btg1, ntr1, ttn1)
                elif nsts1[i, ix] > 0:  # "close" point is updated
                    new_TT = update(veln1, velpn1, vel_map1, nsts1, ttn1, i, ix, dnx1, nnz1, nnx1, phase_vel, stif_den1)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts1, ttn1, dnx1, dnx1, nnx1, nnz1, veln1, velpn1, vel_map1, avlist2, stif_den1)
                    ttn1[i, ix] = new_TT
                    nsts1, btg1 = updtree(i, ix, nsts1, btg1, ttn1)
            elif abs(isz_1 - i) == max_dist1 + 1:
                finished = True


    #plt.imshow(ttn1)
    #plt.gca().invert_yaxis()
    #plt.title("ttn1")
    #plt.show()

    # Set up variables for coarser grid.
    size2 = size1 + 3 * subgrid_size #size2 = 6  # Size of grid in each direction around source for propagation (total is 2n+1)
    subgrid_size2 = 3  # Size of subgrid (must be subgrid_size / 3)
    side2 = int((subgrid_size2 - 1) / 2)
    left = max(0, isx - size2)
    right = min(nnx - 1, isx + size2)
    bottom = max(0, isz - size2)
    top = min(nnz - 1, isz + size2)
    temp_isx = isx - left
    temp_isz = isz - bottom
    veln2 = finer_grid_n(veln[bottom:top+1, left:right+1], subgrid_size2)
    velpn2 = finer_grid_n(velpn[bottom:top+1, left:right+1], subgrid_size2)
    vel_map2 = finer_grid_n(vel_map[bottom:top + 1, left:right + 1], subgrid_size2, numba.float32)

    if stif_den is None:
        stif_den2 = None
    else:
        stif_den2 = finer_grid_n_2(stif_den[bottom:top + 1, left:right + 1], subgrid_size2)

    ttn2 = np.zeros(veln2.shape)
    nsts2 = - np.ones_like(veln2)
    isx_2 = subgrid_size2 * temp_isx
    isz_2 = subgrid_size2 * temp_isz
    dnx2 = dnx / subgrid_size2
    nnz2 = veln2.shape[0]
    nnx2 = veln2.shape[1]
    max_dist2 = subgrid_size2 * size2  # Maximum distance along horiz and vert

    # Set up min heap.
    snb = 0.25
    maxbt = round(snb * veln2.shape[0] * veln2.shape[1])
    #btg2 = np.zeros((maxbt, 2), dtype=int)
    btg2 = np.zeros((maxbt, 2), dtype=numba.int32)
    ntr2 = 0

    # Add points with travel times to the coarser grid and check if the point is next to a points with no travel time.
    # If it is we add the point to the binary tree so neighbouring point can get travel time estimates.
    for i in range(0, ttn1.shape[0] + 1, 3):
        for j in range(0, ttn1.shape[1] + 1, 3):
            pos_z = int(isz_2 + (i - isz_1) / 3)
            pos_x = int(isx_2 + (j - isx_1) / 3)
            ttn2[pos_z, pos_x] = ttn1[i, j]
            if nsts1[i, j] == 0:
                nsts2[pos_z, pos_x] = 0
                outer_point = False
                if i - 3 >= 0:
                    if nsts1[i - 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if i + 3 <= nnz1 - 1:
                    if nsts1[i + 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j - 3 >= 0:
                    if nsts1[i, j - 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j + 3 <= nnx1 - 1:
                    if nsts1[i, j + 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if outer_point == True:
                    nsts2, btg2, ntr2 = addtree(pos_z, pos_x, nsts2, btg2, ntr2, ttn2)
                #if nsts1[i - 3, j] == -1 or nsts1[i + 3, j] != -1 or nsts1[i, j - 3] == -1 or nsts1[i, j + 3] == -1:
                #    nsts2, btg2, ntr2 = addtree(pos_z, pos_x, nsts2, btg2, ntr2, ttn2)

            if nsts1[i, j] > 0:
                nsts2, btg2, ntr2 = addtree(pos_z, pos_x, nsts2, btg2, ntr2, ttn2)
            #fix_val = True
            #if nsts1[i, j] == 0:
            #    if i != 0:
            #        if nsts1[pos_z - 3, pos_x] == -1:
            #            fix_val = False
            #    if i != nnz1 - 1:
            #        if nsts1[pos_z + 3, pos_x] == -1:
            #            fix_val = False
            #    if j != 0:
            #        if nsts1[pos_z, pos_x - 3] == -1:
            #            fix_val = False
            #    if j != nnx1 - 1:
            #        if nsts1[pos_z, pos_x + 3] == -1:
            #            fix_val = False
            #    if fix_val == True:
            #        nsts2[pos_z, pos_x] = 0
            #elif nsts1[i, j] != -1 or fix_val == False:
            #    nsts2, btg2, ntr2 = addtree(i, j, nsts2, btg2, ntr2, ttn2)

    #plt.figure(figsize=(8, 8))
    #plt.imshow(ttn2)
    #plt.gca().invert_yaxis()
    #plt.title("ttn2")
    #plt.draw()
    #plt.show()

    #plt.figure(figsize=(8, 8))
    #plt.imshow(nsts2, vmax=1)
    #plt.gca().invert_yaxis()
    #plt.title("nsts2")
    #plt.show()

    # Update/Find travel time estimates until we reach the edge of the grid.
    finished = False
    while ntr2 > 0 and finished == False:
        # Set the "close" point with minimum traveltime to "alive"
        ix = btg2[1, 1]  # [1,?] as position 1 on binary tree not using 0 index (makes calculating parent and child nodes more complex
        iz = btg2[1, 0]
        nsts2[iz, ix] = 0
        # Update the binary tree by removing the root and sweeping down the tree.
        nsts2, btg2, ntr2, ttn2 = downtree(nsts2, btg2, ntr2, ttn2)

        # Now update or find values of up to four grid points that surround the new "alive" point.
        # Test points that vary in x
        for i in [ix - 1, ix + 1]:
            if 0 <= i <= nnx2 - 1:
                if nsts2[iz, i] == -1:  # "far" point is added to the list of "close" points
                    new_TT = update(veln2, velpn2, vel_map2, nsts2, ttn2, iz, i, dnx2, nnz2, nnx2, phase_vel, stif_den2)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts2, ttn2, dnx2, dnx2, nnx2, nnz2, veln2, velpn2, vel_map2, avlist2, stif_den2)
                    ttn2[iz, i] = new_TT
                    nsts2, btg2, ntr2 = addtree(iz, i, nsts2, btg2, ntr2, ttn2)
                elif nsts2[iz, i] > 0:  # "close" point is updated
                    new_TT = update(veln2, velpn2, vel_map2, nsts2, ttn2, iz, i, dnx2, nnz2, nnx2, phase_vel, stif_den2)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts2, ttn2, dnx2, dnx2, nnx2, nnz2, veln2, velpn2, vel_map2, avlist2, stif_den2)
                    ttn2[iz, i] = new_TT
                    nsts2, btg2 = updtree(iz, i, nsts2, btg2, ttn2)
            elif abs(isx_2 - i) == max_dist2 + 1:
                finished = True


        # Test points that vary in z
        for i in [iz - 1, iz + 1]:  # i=iz-1:2:iz+1
            if 0 <= i <= nnz2 - 1:
                if nsts2[i, ix] == -1:  # "far" point is added to the list of "close" points
                    new_TT = update(veln2, velpn2, vel_map2, nsts2, ttn2, i, ix, dnx2, nnz2, nnx2, phase_vel, stif_den2)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts2, ttn2, dnx2, dnx2, nnx2, nnz2, veln2, velpn2, vel_map2, avlist2, stif_den2)
                    ttn2[i, ix] = new_TT
                    nsts2, btg2, ntr2 = addtree(i, ix, nsts2, btg2, ntr2, ttn2)
                elif nsts2[i, ix] > 0:  # "close" point is updated
                    new_TT = update(veln2, velpn2, vel_map2, nsts2, ttn2, i, ix, dnx2, nnz2, nnx2, phase_vel, stif_den2)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts2, ttn2, dnx2, dnx2, nnx2, nnz2, veln2, velpn2, vel_map2, avlist2, stif_den2)
                    ttn2[i, ix] = new_TT
                    nsts2, btg2 = updtree(i, ix, nsts2, btg2, ttn2)
            elif abs(isz_2 - i) == max_dist2 + 1:
                finished = True

    #plt.imshow(ttn2)
    #plt.title("ttn2")
    #plt.gca().invert_yaxis()
    #plt.show()


    # Unused since a finer grid was already applied at the begining.
    """
    size3 = 13  # Size of grid in each direction around source for propagation (total is 2n+1)
    subgrid_size3 = 3  # Size of subgrid (must be subgrid_size / 3)
    side = int((subgrid_size3 - 1) / 2)
    left = max(0, isx - size3)
    right = min(nnx - 1, isx + size3)
    bottom = max(0, isz - size3)
    top = min(nnz - 1, isz + size3)
    temp_isx = isx - left
    temp_isz = isz - bottom
    veln3 = finer_grid_n(veln[bottom:top+1, left:right+1], subgrid_size3)
    velpn3 = finer_grid_n(velpn[bottom:top+1, left:right+1], subgrid_size3)
    vel_map3 = finer_grid_n(vel_map[bottom:top + 1, left:right + 1], subgrid_size3, numba.float32)
    if type(stif_den) != type(None):
        stif_den3 = finer_grid_n_2(stif_den[bottom:top + 1, left:right + 1], subgrid_size3)
    else:
        stif_den3 = None
    ttn3 = np.zeros(veln3.shape)
    nsts3 = - np.ones_like(veln3)
    isx_3 = subgrid_size3 * temp_isx
    isz_3 = subgrid_size3 * temp_isz
    dnx3 = dnx / subgrid_size3
    dnz3 = dnz / subgrid_size3
    nnz3 = veln3.shape[0]
    nnx3 = veln3.shape[1]
    max_dist3 = subgrid_size3 * size3  # Maximum distance along horiz and vert

    snb = 0.5
    maxbt = round(snb * veln3.shape[0] * veln3.shape[1])
    # btg = np.zeros((maxbt, 2), dtype=numba.int32)
    btg3 = np.zeros((maxbt, 2), dtype=int)
    ntr3 = 0
    sten_no = 0

    for i in range(0, ttn2.shape[0] + 1, 3):
        for j in range(0, ttn2.shape[1] + 1, 3):
            pos_z = int(isz_3 + (i - isz_2) / 3)
            pos_x = int(isx_3 + (j - isx_2) / 3)
            ttn3[pos_z, pos_x] = ttn2[i, j]
            if nsts2[i, j] == 0:
                nsts3[pos_z, pos_x] = 0
                outer_point = False
                if i - 3 >= 0:
                    if nsts2[i - 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if i + 3 <= nnz2 - 1:
                    if nsts2[i + 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j - 3 >= 0:
                    if nsts2[i, j - 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j + 3 <= nnx2 - 1:
                    if nsts2[i, j + 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if outer_point == True:
                    nsts3, btg3, ntr3 = addtree(pos_z, pos_x, nsts3, btg3, ntr3, ttn3)
                # if nsts1[i - 3, j] == -1 or nsts1[i + 3, j] != -1 or nsts1[i, j - 3] == -1 or nsts1[i, j + 3] == -1:
                #    nsts2, btg2, ntr2 = addtree(pos_z, pos_x, nsts2, btg2, ntr2, ttn2)

            if nsts2[i, j] > 0:
                nsts3, btg3, ntr3 = addtree(pos_z, pos_x, nsts3, btg3, ntr3, ttn3)

    #plt.figure(figsize=(8, 8))
    #plt.imshow(ttn3)
    #plt.gca().invert_yaxis()
    #plt.title("ttn3")
    #plt.draw()

    #plt.figure(figsize=(8, 8))
    #plt.imshow(nsts3, vmax=1)
    #plt.gca().invert_yaxis()
    #plt.title("nsts3")
    #plt.show()

    finished = False
    while ntr3 > 0 and finished == False:
        # if math.sqrt(points) % 1 == 0:
        #    print(points)
        #    plt.imshow(ttn)
        #    plt.draw()
        #    plt.pause(0.01)
        #    plt.clf()
        # plt.close()
        # ! Set the "close" point with minimum traveltime
        # ! to "alive"
        # !
        ix = btg3[1, 1]  # [1,?] as position 1 on binary tree not using 0 index (makes calculating parent and
        # child nodes more complex
        iz = btg3[1, 0]
        nsts3[iz, ix] = 0
        # ! Update the binary tree by removing the root and
        # ! sweeping down the tree.
        # !
        nsts3, btg3, ntr3, ttn3 = downtree(nsts3, btg3, ntr3, ttn3)

        # ! Now update or find values of up to four grid points
        # ! that surround the new "alive" point.
        # !
        # ! Test points that vary in x
        # !
        for i in [ix - 1, ix + 1]:  # =ix-1:2:ix+1
            if 0 <= i <= nnx3 - 1:
                if nsts3[iz, i] == -1:
                    # ! This option occurs when a far point is added to the list of "close" points
                    new_TT = update(veln3, velpn3, vel_map3, nsts3, ttn3, iz, i, dnx3, nnz3, nnx3, phase_vel, stif_den3)
                    # if new_TT == -1.0:
                    #    new_TT = fouds18_TT
                    if new_TT == -1.0:  # or new_TT < fouds18_TT:
                        new_TT = fouds18_A(iz, i, nsts3, ttn3, dnx3, dnx3, nnx3, nnz3, veln3, velpn3, vel_map3, avlist2, stif_den3)
                    ttn3[iz, i] = new_TT
                    # ttn = fouds18(iz, i, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, avlist2)
                    nsts3, btg3, ntr3 = addtree(iz, i, nsts3, btg3, ntr3, ttn3)
                    # plt.imshow(ttn)
                    # plt.show()
                elif nsts3[iz, i] > 0:
                    # !
                    # ! This happens when a "close" point is updated
                    # !
                    new_TT = update(veln3, velpn3, vel_map3, nsts3, ttn3, iz, i, dnx3, nnz3, nnx3, phase_vel, stif_den3)
                    # if new_TT == -1.0:
                    #    new_TT = fouds18_TT
                    if new_TT == -1.0:  # or new_TT < fouds18_TT:
                        new_TT = fouds18_A(iz, i, nsts3, ttn3, dnx3, dnx3, nnx3, nnz3, veln3, velpn3, vel_map3, avlist2, stif_den3)
                    ttn3[iz, i] = new_TT
                    # ttn = fouds18(iz, i, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, avlist2)
                    nsts3, btg3 = updtree(iz, i, nsts3, btg3, ttn3)
            elif abs(isx_3 - i) == max_dist3 + 1:
                finished = True

        # !
        # ! Test points that vary in z
        # !
        for i in [iz - 1, iz + 1]:  # i=iz-1:2:iz+1
            if 0 <= i <= nnz3 - 1:
                if nsts3[i, ix] == -1:
                    # ! This option occurs when a far point is added to the list of "close" points           % !
                    new_TT = update(veln3, velpn3, vel_map3, nsts3, ttn3, i, ix, dnx3, nnz3, nnx3, phase_vel, stif_den3)
                    if new_TT == -1.0:  # or new_TT < fouds18_TT:
                        new_TT = fouds18_A(i, ix, nsts3, ttn3, dnx3, dnx3, nnx3, nnz3, veln3, velpn3, vel_map3, avlist2, stif_den3)
                    ttn3[i, ix] = new_TT
                    # plt.imshow(ttn)
                    # plt.show()
                    # ttn = fouds18(i, ix, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, avlist2)
                    nsts3, btg3, ntr3 = addtree(i, ix, nsts3, btg3, ntr3, ttn3)
                elif nsts3[i, ix] > 0:
                    # !
                    # ! This happens when a "close" point is updated
                    # !
                    new_TT = update(veln3, velpn3, vel_map3, nsts3, ttn3, i, ix, dnx3, nnz3, nnx3, phase_vel, stif_den3)
                    if new_TT == -1.0:  # or new_TT < fouds18_TT:
                        new_TT = fouds18_A(i, ix, nsts3, ttn3, dnx3, dnx3, nnx3, nnz3, veln3, velpn3, vel_map3, avlist2, stif_den3)
                    ttn3[i, ix] = new_TT
                    # ttn = fouds18(i, ix, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, avlist2)
                    nsts3, btg3 = updtree(i, ix, nsts3, btg3, ttn3)
            elif abs(isz_3 - i) == max_dist3 + 1:
                finished = True

    #plt.figure(figsize=(8, 8))
    #plt.imshow(ttn3)
    #plt.gca().invert_yaxis()
    #plt.title("ttn3")
    #plt.show()

    nsts = - np.ones(veln.shape, dtype=int)
    for i in range(0, ttn3.shape[0] + 1, 3):
        for j in range(0, ttn3.shape[1] + 1, 3):
            pos_z = int(isz + (i - isz_3) / 3)
            pos_x = int(isx + (j - isx_3) / 3)
            ttn[pos_z, pos_x] = ttn3[i, j]
            if nsts3[i, j] == 0:
                nsts[pos_z, pos_x] = 0
                outer_point = False
                if i - 3 >= 0:
                    if nsts3[i - 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if i + 3 <= nnz3 - 1:
                    if nsts3[i + 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j - 3 >= 0:
                    if nsts3[i, j - 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j + 3 <= nnx3 - 1:
                    if nsts3[i, j + 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if outer_point == True:
                    nsts, btg, ntr = addtree(pos_z, pos_x, nsts, btg, ntr, ttn)
                #if nsts1[i - 3, j] == -1 or nsts1[i + 3, j] != -1 or nsts1[i, j - 3] == -1 or nsts1[i, j + 3] == -1:
                #    nsts2, btg2, ntr2 = addtree(pos_z, pos_x, nsts2, btg2, ntr2, ttn2)

            if nsts3[i, j] > 0:
                nsts, btg, ntr = addtree(pos_z, pos_x, nsts, btg, ntr, ttn)
    """

    # Set up variables for the original grid and move points with a travel time in the previous grid that have a corresponding point in the new grid to the new grid.
    # Check if these points have neighbouring points without a travel time and if they do, add them to the min heap so neighbouring points can be updated.
    nsts = - np.ones_like(veln)
    for i in range(0, ttn2.shape[0] + 1, 3):
        for j in range(0, ttn2.shape[1] + 1, 3):
            pos_z = int(isz + (i - isz_2) / 3)
            pos_x = int(isx + (j - isx_2) / 3)
            ttn[pos_z, pos_x] = ttn2[i, j]
            if nsts2[i, j] == 0:
                nsts[pos_z, pos_x] = 0
                outer_point = False
                if i - 3 >= 0:
                    if nsts2[i - 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if i + 3 <= nnz2 - 1:
                    if nsts2[i + 3, j] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j - 3 >= 0:
                    if nsts2[i, j - 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if j + 3 <= nnx2 - 1:
                    if nsts2[i, j + 3] == -1:
                        outer_point = True
                else:
                    outer_point = True
                if outer_point == True:
                    nsts, btg, ntr = addtree(pos_z, pos_x, nsts, btg, ntr, ttn)
                # if nsts1[i - 3, j] == -1 or nsts1[i + 3, j] != -1 or nsts1[i, j - 3] == -1 or nsts1[i, j + 3] == -1:
                #    nsts2, btg2, ntr2 = addtree(pos_z, pos_x, nsts2, btg2, ntr2, ttn2)

            if nsts2[i, j] > 0:
                nsts, btg, ntr = addtree(pos_z, pos_x, nsts, btg, ntr, ttn)

    #plt.figure(figsize=(8, 8))
    #plt.imshow(ttn)
    #plt.gca().invert_yaxis()
    #plt.title("ttn")
    #plt.show()
    #plt.draw()

    #plt.figure(figsize=(8, 8))
    #plt.imshow(nsts, vmax=1)
    #plt.gca().invert_yaxis()
    #plt.title("nsts")
    #plt.show()

    # Update/find travel times until all points in the grid have a travel time.
    while ntr > 0:
        # Set the "close" point with minimum travel time to "alive"
        ix = btg[1, 1]  # [1,?] as position 1 on binary tree not using 0 index (makes calculating parent and child nodes more complex
        iz = btg[1, 0]
        nsts[iz, ix] = 0
        # Update the binary tree by removing the root and sweeping down the tree.
        nsts, btg, ntr, ttn = downtree(nsts, btg, ntr, ttn)

        # Now update or find values of up to four grid points that surround the new "alive" point.
        # Test points that vary in x
        for i in [ix - 1, ix + 1]:
            if 0 <= i <= nnx - 1:
                if nsts[iz, i] == -1:  # "far" point is added to the list of "close" points.
                    new_TT = update(veln, velpn, vel_map, nsts, ttn, iz, i, dnx, nnz, nnx, phase_vel, stif_den)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, vel_map, avlist2, stif_den)
                    ttn[iz, i] = new_TT
                    nsts, btg, ntr = addtree(iz, i, nsts, btg, ntr, ttn)
                    #plt.imshow(ttn)
                    #plt.show()
                elif nsts[iz, i] > 0:  # "close" point is updated.
                    new_TT = update(veln, velpn, vel_map, nsts, ttn, iz, i, dnx, nnz, nnx, phase_vel, stif_den)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(iz, i, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, vel_map, avlist2, stif_den)
                    ttn[iz, i] = new_TT
                    nsts, btg = updtree(iz, i, nsts, btg, ttn)


        # Test points that vary in z.
        for i in [iz - 1, iz + 1]:  # i=iz-1:2:iz+1
            if 0 <= i <= nnz - 1:
                if nsts[i, ix] == -1:  # "far" point is added to the list of "close" points
                    new_TT = update(veln, velpn, vel_map, nsts, ttn, i, ix, dnx, nnz, nnx, phase_vel, stif_den)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, vel_map, avlist2, stif_den)
                    ttn[i, ix] = new_TT
                    nsts, btg, ntr = addtree(i, ix, nsts, btg, ntr, ttn)
                elif nsts[i, ix] > 0:  # "close" point is updated
                    new_TT = update(veln, velpn, vel_map, nsts, ttn, i, ix, dnx, nnz, nnx, phase_vel, stif_den)
                    if new_TT == -1.0:  # If no stencil could be used.
                        new_TT = fouds18_A(i, ix, nsts, ttn, dnx, dnz, nnx, nnz, veln, velpn, vel_map, avlist2, stif_den)
                    ttn[i, ix] = new_TT
                    nsts, btg = updtree(i, ix, nsts, btg, ttn)

    #plt.figure(figsize=(8, 8))
    #plt.imshow(ttn)
    #plt.gca().invert_yaxis()
    #plt.title("ttn")
    #plt.show()
    #plt.draw()

    #plt.figure(figsize=(8, 8))
    #plt.contourf(ttn, 30)
    #plt.gca().invert_yaxis()
    #plt.title("ttn")
    #plt.show()

    return ttn / subgrid_size


@njit(cache=True)
def time_between_points(x1, x2, y1, y2, dnx, subgrid_size, velocity_dat, veln, velpn, vel_map, stif_den):
    """
    Finds the travel time of a straight ray between two points on a finer grid.

    :param x1: x index of the starting position on the finer grid. Does not need to be an integer value.
    :type x1: float
    :param x2: x index of the end position on the finer grid. Does not need to be an integer value.
    :type x2: float
    :param y1: y index of the starting position on the finer grid. Does not need to be an integer value.
    :type y1: float
    :param y2: y index of the end position on the finer grid. Does not need to be an integer value.
    :type y2: float
    :param dnx: Distance between grid points on the original grid.
    :type dnx: float
    :param subgrid_size: Odd integer for the size that the original grid was increased by to obtain the finer grid.
    :type subgrid_size: int
    :param velocity_dat: Group velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type velocity_dat: 2D numpy array
    :param veln: Anisotropic orientation of all grid points on the original grid.
    :type veln: 2D numpy array
    :param velpn: Material index of all grid points on the original grid (0 if using stiffness tensors and density, otherwise index for column in avlist2).
    :type velpn: 2D numpy array of type int
    :param vel_map: Value used for scaling velocities at all grid points on the original grid (mainly used for isotropic materials).
    :type vel_map: 2D numpy array
    :param stif_den: Material parameters with first two indices being the i,j coordinates and the third being the index of the material parameter(c_22, c_23, c_33, c_44, density). Stiffness tensors must be in MPa to avoid overflow errors.
    :type stif_den: 3D numpy array of type int64
    :return: Travel time of straight ray.
    :rtype: float
    """

    # Find the coordinates on the original grid.
    x1 = x1 / subgrid_size
    x2 = x2 / subgrid_size
    y1 = y1 / subgrid_size
    y2 = y2 / subgrid_size

    # Set up variables for keeping track of the position of the ray and total time.
    section_time = 0.0
    start_x = x1
    end_x = x2
    start_y = y1
    end_y = y2
    prev_x = x1
    prev_y = y1

    # Find the direction of the ray.
    if x1 == x2:
        angle = 0
    else:
        angle = math.degrees(math.atan((y2 - y1) / (x2 - x1)))
    #total_length = dnx * math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / subgrid_size

    # Find the equation of the ray path as y=mx+c
    if end_x != start_x:
        m = (end_y - start_y) / (end_x - start_x)
        c = start_y - m * start_x
    finished_x = False
    finished_y = False

    # determine if x and y values increase or decrease along the ray path.
    if start_x < end_x:
        dir_x = 1
    else:
        dir_x = -1
    if start_y < end_y:
        dir_y = 1
    else:
        dir_y = -1

    # Find the next x and y values where the ray path will intersect the boundary between two points.
    next_x = round(start_x) + dir_x * 0.5
    next_y = round(start_y) + dir_y * 0.5

    # Go through all cells that the ray path passes and determine the travel time in that section of the ray.
    while not (finished_x and finished_y):
        # Determine if the next cell boundary in the x and y directions is after the rnd of the ray.
        if ((next_x > end_x and dir_x == 1) or (next_x < end_x and dir_x == -1)) and finished_x == False:
            finished_x = True
            next_x = end_x
        if ((next_y > end_y and dir_y == 1) or (next_y < end_y and dir_y == -1)) and finished_y == False:
            finished_y = True
            next_y = end_y
        # Determine coordinates of next points of intersection.
        if end_x == start_x:
            next_x_val = start_x
            next_y_val = next_y
            next_y += dir_y
        else:
            next_x_yval = m * next_x + c
            if m != 0:
                next_y_xval = (next_y - c) / m
                if (start_x - next_x) ** 2 + (start_y - next_x_yval) ** 2 < (start_x - next_y_xval) ** 2 + (start_y - next_y) ** 2:
                    next_x_val = next_x
                    next_y_val = next_x_yval
                    next_x += dir_x
                else:
                    next_x_val = next_y_xval
                    next_y_val = next_y
                    next_y += dir_y
            else:
                next_x_val = next_x
                next_y_val = next_x_yval
                next_x += dir_x
        x_pos = round((prev_x + next_x_val) / 2)
        y_pos = round((prev_y + next_y_val) / 2)
        # Determine the effective angle of the ray path through the material in the region.
        eff_ang = (veln[y_pos, x_pos] - angle) % 180
        distance = dnx * math.sqrt((prev_x - next_x_val) ** 2 + (prev_y - next_y_val) ** 2)
        #if x1 != x2:
        #    distance = total_length * (prev_x - next_x_val)/(x1 - x2)
        #else:
        #    distance = total_length * (prev_y - next_y_val)/(y1 - y2)

        # Determine the velocity of the ray in the section.
        if velpn[y_pos, x_pos] != 0 or stif_den == None:
            angle1 = math.floor(eff_ang)
            angle2 = (angle1 + 1) % 180
            remainder = eff_ang - angle1
            velocity = vel_map[y_pos, x_pos] * ((1 - remainder) * velocity_dat[angle1, velpn[y_pos, x_pos]] + remainder * velocity_dat[angle2, velpn[y_pos, x_pos]])
        else:
            # Solve the christoffel equation for group velocity.
            sigma = stif_den[y_pos, x_pos, 4]
            if eff_ang % 90 < 0.01 or eff_ang % 90 > 90 - 0.01:
                if abs((eff_ang % 180) - 90) < 1:
                    lambda_val = stif_den[y_pos, x_pos, 2]
                else:
                    lambda_val = stif_den[y_pos, x_pos, 0]
                velocity = 1000 * vel_map[y_pos, x_pos] * math.sqrt(lambda_val / sigma)
            else:
                c_22 = stif_den[y_pos, x_pos, 0]
                c_23 = stif_den[y_pos, x_pos, 1]
                c_33 = stif_den[y_pos, x_pos, 2]
                c_44 = stif_den[y_pos, x_pos, 3]
                tan_ang = math.tan(math.radians(eff_ang))
                A = c_22 + c_33 - 2 * c_44
                B = (c_23 + c_44) * (tan_ang - 1 / tan_ang)
                C = c_22 - c_33
                if eff_ang < 90:
                    phase_angle_rad = math.atan((-B - math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
                else:
                    phase_angle_rad = math.atan((-B + math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
                lambda_val = 0.5 * (math.cos(2 * phase_angle_rad) * (c_22 - c_44) + math.sin(2 * phase_angle_rad) * (c_23 + c_44) * tan_ang + c_22 + c_44)
                velocity = 1000 * vel_map[y_pos, x_pos] * math.sqrt(lambda_val / sigma) / math.cos(math.radians(eff_ang) - phase_angle_rad)

        # Update the total time with the last section of the ray.
        slown = 1.0 / velocity
        section_time += distance * slown

        # Move to new section.
        prev_x = next_x_val
        prev_y = next_y_val
    # print(section_time)
    # input("done")
    return section_time


@njit(cache=True)
def ray_time(ray_x, ray_y, dnx, subgrid_size, velocity_dat, veln, velpn, vel_map, stif_den):
    """
    Determine the travel time along a ray path by integrating along the path.

    :param ray_x: x indices of the points in the ray path on the finer grid.
    :type ray_x: 1D numpy array
    :param ray_y: y indices of the points in the ray path on the finer grid.
    :type ray_y: 1D numpy array
    :param dnx: Distance between points on the original grid.
    :type dnx: float
    :param subgrid_size: Size that the original grid was increased by to obtain the finer grid (odd integer)
    :type subgrid_size: int
    :param velocity_dat: Group velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type velocity_dat: 2D numpy array
    :param veln: Anisotropic orientation of all grid points on the original grid.
    :type veln: 2D numpy array
    :param velpn: Material index of all grid points on the original grid (0 if using stiffness tensors and density, otherwise index for column in avlist2).
    :type velpn: 2D numpy array of type int
    :param vel_map: Value used for scaling velocities at all grid points on the original grid (mainly used for isotropic materials).
    :type vel_map: 2D numpy array
    :param stif_den: 3D numpy array of type int64 of material parameters with first two indices being the i,j coordinates and the third being the index of the material parameter(c_22, c_23, c_33, c_44, density). Stiffness tensors must be in MPa to avoid overflow errors.
    :type stif_den: 3D numpy array of type int64
    :return: Travel time along ray path
    :rtype: float
    """
    # Loop through all sections of the ray path and sum all the travel times of the sections.
    trav_time = 0.0
    for i in range(len(ray_x) - 1):
        trav_time += time_between_points(ray_x[i], ray_x[i + 1], ray_y[i], ray_y[i + 1], dnx, subgrid_size, velocity_dat, veln, velpn, vel_map, stif_den)
    return trav_time


@njit(cache=True)
def travel_times(ray_x, ray_y):
    """
    Splits a ray path up into smaller sections such that we have a point every time we change direction or change between grid cells.

    :param ray_x: x indices of the points in the ray path.
    :type ray_x: 1D numpy array
    :param ray_y: y indices of the points in the ray path.
    :type ray_y: 1D numpy array
    :return: ray_x, ray_y - arrays of the x and y indices of ray path which has been split up.
    :rtype: 2D numpy array, 2D numpy array
    """
    ray_x1 = np.array([ray_x[0]])
    ray_y1 = np.array([ray_y[0]])

    # Go through each section of the ray a split it up into smaller sections.
    for i in range(len(ray_x) - 1):
        start_x = ray_x[i]
        end_x = ray_x[i + 1]
        start_y = ray_y[i]
        end_y = ray_y[i + 1]

        if end_x != start_x:
            m = (end_y - start_y) / (end_x - start_x)
            c = start_y - m * start_x
        finished_x = False
        finished_y = False
        if start_x < end_x:
            dir_x = 1
        else:
            dir_x = -1
        if start_y < end_y:
            dir_y = 1
        else:
            dir_y = -1
        # Find where the ray path next crosses a boundary in the x and y direction.
        next_x = round(start_x) + dir_x * 0.5
        next_y = round(start_y) + dir_y * 0.5
        # Loop through all regions that the ray path crosses.
        while not (finished_x and finished_y):
            if (next_x > end_x and dir_x == 1) or (next_x < end_x and dir_x == -1):
                finished_x = True
                next_x = end_x
            if (next_y > end_y and dir_y == 1) or (next_y < end_y and dir_y == -1):
                finished_y = True
                next_y = end_y
            if end_x == start_x:
                next_x_val = start_x
                next_y_val = next_y
                next_y += dir_y
            else:
                next_x_yval = m * next_x + c
                if m != 0:
                    next_y_xval = (next_y - c) / m
                    if (start_x - next_x) ** 2 + (start_y - next_x_yval) ** 2 < (
                            start_x - next_y_xval) ** 2 + (
                            start_y - next_y) ** 2:
                        next_x_val = next_x
                        next_y_val = next_x_yval
                        next_x += dir_x
                    else:
                        next_x_val = next_y_xval
                        next_y_val = next_y
                        next_y += dir_y
                else:
                    next_x_val = next_x
                    next_y_val = next_x_yval
                    next_x += dir_x

            # Add points onto the ray path.
            ray_x1 = np.append(ray_x1, next_x_val)
            ray_y1 = np.append(ray_y1, next_y_val)
            # plt.plot(ray_x, ray_y, "bx")
            # plt.plot(ray_x1, ray_y1, "r+")
            # plt.grid()
            # plt.show()
    return ray_x1, ray_y1


@njit(cache=True)
def find_ray(dnx, velocity_dat, source, receiver, rec_TTF, veln, velpn, vel_map, stif_den, subgrid_size):
    """
    Finds a ray path using the travel time field with the receiver as the source. The travel time field is calculated on a finer grid than the original grid.

    :param dnx: Distance between points on the original grid.
    :type dnx: float
    :param velocity_dat: Group velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type velocity_dat: 2D numpy array
    :param source: indices for the coordinates of the source on the original grid i.e [i, j].
    :type source: array of length 2
    :param receiver: indices of the coordinates of the receiver on the original grid.
    :type receiver: array of length 2
    :param rec_TTF: Travel time field with the receiver as the source on the finer grid.
    :type rec_TTF: 2D numpy array
    :param veln: 2D numpy array for the anisotropic orientation of all grid points on the original grid.
    :type veln: 2D numpy array
    :param velpn: Material index of all grid points on the original grid(0 if using stiffness tensors and density, otherwise index for column in avlist2).
    :type velpn: 2D numpy array of type int
    :param vel_map: Value used for scaling velocities at all grid points on the original grid (mainly used for isotropic materials).
    :type vel_map: 2D numpy array
    :param stif_den: Material parameters with first two indices being the i,j coordinates and the third being the index of the material parameter(c_22, c_23, c_33, c_44, density). Stiffness tensors must be in MPa to avoid overflow errors.
    :type stif_den: 3D numpy array of type int64
    :param subgrid_size: The size increase from the original grid to the finer grid.
    :type subgrid_size: int
    :return: ray_x, ray_y, travel_time - Arrays for the x and y indices of the points on the ray path and the travel time along the ray.
    :rtype: 1D numpy array, 1D numpy array, float
    """
    # Parameter for distance along plane for center.
    plane_dist = 3
    search_dist = plane_dist * subgrid_size + 1
    #search_dist_2 = plane_dist * subgrid_size - int((subgrid_size - 1) / 2)
    search_dist_2 = (plane_dist - 1) * subgrid_size + 1


    # Set up arrays for storing the ray path.
    ray_x = np.zeros(5 * (veln.shape[0] + veln.shape[1]))  # number of points in ray undetermined so setting a max length
    ray_y = np.copy(ray_x)
    ray_x[0] = source[0]
    ray_y[0] = source[1]
    last_x = source[0]
    last_y = source[1]
    ray_len = 1
    # Set the last section of the ray to the vector from source to reciever to get an initial direction to determine the plane to use for ray tracing.
    last_vect_x = receiver[0] - source[0]
    last_vect_y = receiver[1] - source[1]
    c_value = 0
    nnx = rec_TTF.shape[0]
    nnz = rec_TTF.shape[1]

    # Add points to the ray path untill we are sufficiently close to the reciever to use a stright ray to the reciever (end of the ray).
    # We choose 1.6 arbitrarily so that we do not go past the reciever and the distance between the last two points is not too large.
    while (last_x - receiver[0]) ** 2 + (last_y - receiver[1]) ** 2 > (1.6 * subgrid_size) ** 2:
        # Ensures ray goes to receiver when close.
        if (last_x - receiver[0]) ** 2 + (last_y - receiver[1]) ** 2 < (4 * subgrid_size) ** 2:
            last_vect_x = receiver[0] - last_x
            last_vect_y = receiver[1] - last_y
        # computes max value of absolute value of dot product between last section of the ray path and unit vectors of [1,0],[1,1],[0,1] and [1,-1] to determine which plane to use in ray tracing.
        # Absolute value is used to account for two directions at once (vector rotated 180deg as well).
        dir_index = np.argmax(np.array([abs(last_vect_x), abs(last_vect_x + last_vect_y) / math.sqrt(2), abs(last_vect_y), abs(last_vect_x - last_vect_y) / math.sqrt(2)]))
        if dir_index == 0:
            # Plane is in the form x = c so we find the c value of the closest point (integer values ensure the plane passes through points in the grid.
            c_value = round(last_x)
            # Determine which of the two directions should be used
            if last_vect_x > 0:  # plane is to the right of the last point in the ray
                c_value += subgrid_size
            else:  # moving to left
                c_value -= subgrid_size
            if c_value < 0 or c_value >= nnz:
                break
            #if 0 < round(last_x) < nnx - 1:
            #    if rec_TTF[round(last_y), round(last_x) - 1] > rec_TTF[round(last_y), round(last_x) + 1]:
            #        c_value += subgrid_size
            #    else:  # moving to left
            #        c_value -= subgrid_size
            #elif round(last_x) == 0:
            #    c_value += subgrid_size
            #else:
            #    c_value -= subgrid_size
            min_val = max(0, round(last_y) - search_dist)
            max_val = min(nnx - 1, round(last_y) + search_dist)
            TT = np.zeros(max_val - min_val + 1)
            # Find the fastest time from the last point in the ray to the receiver when the ray must go through a point in the plane
            for i in range(len(TT)):
                x_val = i + min_val
                TT[i] = rec_TTF[x_val, c_value] + time_between_points(last_x, c_value, last_y, x_val, dnx, subgrid_size, velocity_dat, veln, velpn, vel_map, stif_den)

            # If there are no local minimums then the minimum over the interval is either at the beginning or the end.
            if TT[0] < TT[len(TT) - 1]:
                minimum = TT[0]
                min_i = 0
            else:
                minimum = TT[len(TT) - 1]
                min_i = len(TT) - 1
            # Look for a local minimum over all possible sets of three successive points.
            for j in range(1, len(TT) - 1):
                time1 = TT[j - 1]
                time2 = TT[j]
                time3 = TT[j + 1]
                if time1 >= time2 <= time3:
                    # If there is a local minimum between the points fit a quadratic equation to the three points and use it to predict the location and value of the local minimum
                    a = (time1 + time3 - 2 * time2) / 2  # quadratic using x_values of -1,0 and 1 (increasing x values by j returns values back to correct position).
                    b = (time3 - time1) / 2
                    c = time2
                    if a != 0:
                        position = - b / (2 * a)
                        local_min_val = a * (position ** 2) + b * position + c
                        position += j
                    else:
                        position = j
                        local_min_val = time2
                    # For finding the global minimum from the local minimums
                    if local_min_val < minimum:
                        min_i = position
                        minimum = local_min_val
            # Add new point to the ray path (length is incremented at the end of the loop).
            ray_x[ray_len] = c_value
            ray_y[ray_len] = min_i + min_val
        elif dir_index == 1:
            # Plane is in the form y = -x+c so we find the c value of the closest point (integer values ensure the plane passes through points in the grid.
            c_value = round(last_x) + round(last_y)
            # Determine which of the two directions we want to use.
            if last_vect_x > 0:  # moving up right
                c_value += subgrid_size
                min_x = max(0, c_value - (nnx - 1), round(last_x) - search_dist_2)
                max_x = min(nnz - 1, c_value, c_value - round(last_y) + search_dist_2)
            else:  # moving down left
                c_value -= subgrid_size
                min_x = max(0, c_value - (nnx - 1), c_value - round(last_y) - search_dist_2)
                max_x = min(nnz - 1, c_value, round(last_x) + search_dist_2)
            #if (0 < round(last_x) < nnx - 1) and (0 < round(last_y) < nnz - 1):
            #    if rec_TTF[round(last_y) - 1, round(last_x) - 1] > rec_TTF[round(last_y) + 1, round(last_x) + 1]:
            #        c_value += subgrid_size
            #    else:  # moving down left
            #        c_value -= subgrid_size
            #elif round(last_x) == 0 or round(last_y) == nnz - 1:
            #    c_value -= subgrid_size
            #else:
            #    c_value += subgrid_size
            # Determine which values of x are valid.
            #min_x = max(0, c_value - (nnx - 1))
            #max_x = min(nnz - 1, c_value)
            TT = np.zeros(max_x - min_x + 1)
            # Find the fastest time from the last point in the ray to the receiver when the ray must go through a point in the plane
            for i in range(len(TT)):
                x_coord = min_x + i
                y_coord = - x_coord + c_value
                #TT[i] = rec_TTF[x_coord, y_coord] + time_between_points(last_x, x_coord, last_y, y_coord, dnx, subgrid_size, velocity_dat, veln, velpn, vel_map, stif_den)
                TT[i] = rec_TTF[y_coord, x_coord] + time_between_points(last_x, x_coord, last_y, y_coord, dnx, subgrid_size, velocity_dat, veln, velpn, vel_map, stif_den)

            # If no local minimum is found then the minimum over the the interval is either at the beginning or the end.
            if TT[0] < TT[len(TT) - 1]:
                minimum = TT[0]
                min_i = 0
            else:
                minimum = TT[len(TT) - 1]
                min_i = len(TT) - 1
            # We check all possible sets of the consecutive points for a local minimum.
            for j in range(1, len(TT) - 1):
                time1 = TT[j - 1]
                time2 = TT[j]
                time3 = TT[j + 1]
                if time1 >= time2 <= time3:
                    # If there is a local minimum between the points fit a quadratic equation to the three points and use it to predict the location and value of the local minimum
                    a = (time1 + time3 - 2 * time2) / 2  # quadratic using x_values of -1,0 and 1 (increasing position value by j returns values back to correct position).
                    b = (time3 - time1) / 2
                    c = time2
                    if a != 0:
                        position = - b / (2 * a)
                        local_min_val = a * (position ** 2) + b * position + c
                        position += j
                    else:
                        position = j
                        local_min_val = time2
                    # For finding global minimum
                    if local_min_val < minimum:
                        min_i = position
                        minimum = local_min_val
            # Add new point to the ray path (length is incremented at the end of the loop).

            ray_x[ray_len] = min_x + min_i
            ray_y[ray_len] = c_value - ray_x[ray_len]
        elif dir_index == 2:
            # Plane is in the form y = c so we find the c value of the closest point (integer values ensure the plane passes through points in the grid.
            c_value = round(last_y)
            # Check which of the two directions we want to use.
            if last_vect_y > 0:  # moving up
                c_value += subgrid_size
            else:
                c_value -= subgrid_size
            if c_value < 0 or c_value >= nnx:
                break
            #if 0 < round(last_y) < nnz - 1:
            #    if rec_TTF[round(last_y) - 1, round(last_x)] > rec_TTF[round(last_y) + 1, round(last_x)]:
            #        c_value += subgrid_size
            #    else:
            #        c_value -= subgrid_size
            #elif round(last_y) == 0:
            #    c_value += subgrid_size
            #else:
            #    c_value -= subgrid_size
            min_val = max(0, round(last_x) - search_dist)
            max_val = min(nnz - 1, round(last_x) + search_dist)

            TT = np.zeros(max_val - min_val + 1)
            # Find the fastest time from the last point in the ray to the receiver when the ray must go through a point in the plane.
            for i in range(len(TT)):
                y_val = i + min_val
                TT[i] = rec_TTF[c_value, y_val] + time_between_points(last_x, y_val, last_y, c_value, dnx, subgrid_size, velocity_dat, veln, velpn, vel_map, stif_den)
            # If no local minimum is found then the minimum over the the interval is either at the beginning or the end.
            if TT[0] < TT[len(TT) - 1]:
                minimum = TT[0]
                min_i = 0
            else:
                minimum = TT[len(TT) - 1]
                min_i = len(TT) - 1
            # We check all possible sets of the consecutive points for a local minimum.
            for j in range(1, len(TT) - 1):
                time1 = TT[j - 1]
                time2 = TT[j]
                time3 = TT[j + 1]
                if time1 >= time2 <= time3:
                    # If there is a local minimum between the points fit a quadratic equation to the three points and use it to predict the location and value of the local minimum.
                    a = (time1 + time3 - 2 * time2) / 2  # quadratic using x_values of -1,0 and 1 (increasing position value by j returns values back to correct position).
                    b = (time3 - time1) / 2
                    c = time2
                    if a != 0:
                        position = - b / (2 * a)
                        local_min_val = a * (position ** 2) + b * position + c
                        position += j
                    else:
                        position = j
                        local_min_val = time2
                    if local_min_val < minimum:
                        min_i = position
                        minimum = local_min_val
            # Add new point to the ray path (length is incremented at the end of the loop).
            ray_x[ray_len] = min_i + min_val
            ray_y[ray_len] = c_value
        else:
            # Plane is in the form y = x+c so we find the c value of the closest point (integer values ensure the plane passes through points in the grid.
            c_value = round(last_y) - round(last_x)
            # Determine which of the two directions to use.
            if last_vect_x < 0:  # moving up left
                c_value += subgrid_size
                min_x = max(0, - c_value, round(last_y) - c_value - search_dist_2)
                max_x = min((nnz - 1), (nnx - 1) - c_value, round(last_x) + search_dist_2)
            else:  # moving down right
                c_value -= subgrid_size
                min_x = max(0, - c_value, round(last_x) - search_dist_2)
                max_x = min((nnz - 1), (nnx - 1) - c_value, round(last_y) - c_value + search_dist_2)
            #if (subgrid_size < round(last_x) < nnx - 1 - subgrid_size) and (0 < round(last_y) < nnz - 1):
            #    print(rec_TTF[round(last_y) - subgrid_size, round(last_x) + subgrid_size], rec_TTF[round(last_y) + subgrid_size, round(last_x) - subgrid_size])
            #    if rec_TTF[round(last_y) - subgrid_size, round(last_x) + subgrid_size] < rec_TTF[round(last_y) + subgrid_size, round(last_x) - subgrid_size]:
            #        c_value += subgrid_size
            #    else:  # moving down right
            #        c_value -= subgrid_size
            #elif round(last_x) < subgrid_size and round(last_y) < subgrid_size:
            #    c_value += subgrid_size
            #else:
            #    c_value -= subgrid_size
            # Determine which values of x are in the grid.
            #min_x = max(0, - c_value)
            #max_x = min((nnz - 1), (nnx - 1) - c_value)
            TT = np.zeros(max_x - min_x + 1)
            # Find the fastest time from the last point in the ray to the receiver when the ray must go through a point in the plane.
            for i in range(len(TT)):
                x_coord = min_x + i
                y_coord = x_coord + c_value
                TT[i] = rec_TTF[y_coord, x_coord] + time_between_points(last_x, x_coord, last_y, y_coord, dnx, subgrid_size, velocity_dat, veln, velpn, vel_map, stif_den)
            # If no local minimum is found then the minimum over the the interval is either at the beginning or the end.
            if TT[0] < TT[len(TT) - 1]:
                minimum = TT[0]
                min_i = 0
            else:
                minimum = TT[len(TT) - 1]
                min_i = len(TT) - 1
            # We check all possible sets of the consecutive points for a local minimum.
            for j in range(1, len(TT) - 1):
                time1 = TT[j - 1]
                time2 = TT[j]
                time3 = TT[j + 1]
                if time1 >= time2 <= time3:
                    # If there is a local minimum between the points fit a quadratic equation to the three points and use it to predict the location and value of the local minimum
                    a = (time1 + time3 - 2 * time2) / 2  # quadratic using x_values of -1,0 and 1 (increasing position value by j returns values back to correct position).
                    b = (time3 - time1) / 2
                    c = time2

                    if a != 0:
                        position = - b / (2 * a)
                        local_min_val = a * (position ** 2) + b * position + c
                        position += j
                    else:
                        position = j
                        local_min_val = time2
                    if local_min_val < minimum:
                        min_i = position
                        minimum = local_min_val
            # Add new point to the ray path (length is incremented at the end of the loop).
            ray_x[ray_len] = min_x + min_i
            ray_y[ray_len] = ray_x[ray_len] + c_value
        # Update the vector for the last section of the ray and update the position of the latest point in the ray path.
        if rec_TTF[round(last_y), round(last_x)] < rec_TTF[round(ray_y[ray_len]), round(ray_x[ray_len])]:
            print("Travel time to receiver increasing: Finishing ray early")
            """
            plt.imshow(veln, interpolation="nearest", cmap="hsv", vmin=0, vmax=90)
            plt.plot(source[0] / subgrid_size, source[1] / subgrid_size, "x", color="orange")
            plt.plot(receiver[0] / subgrid_size, receiver[1] / subgrid_size, "+", color="orange")
            plt.plot(ray_x[0:ray_len + 1] / subgrid_size, ray_y[0:ray_len + 1] / subgrid_size, "k")
            plt.plot(ray_x[0:ray_len + 1] / subgrid_size, ray_y[0:ray_len + 1] / subgrid_size, "kx", markersize=1)
            plt.show()
            plt.contourf(rec_TTF, 100, cmap="flag")
            plt.imshow(veln, interpolation="nearest", cmap="hsv", vmin=0, vmax=90)
            plt.plot(source[0] / subgrid_size, source[1] / subgrid_size, "x", color="orange")
            plt.plot(receiver[0] / subgrid_size, receiver[1] / subgrid_size, "+", color="orange")
            plt.plot(ray_x[0:ray_len + 1] / subgrid_size, ray_y[0:ray_len + 1] / subgrid_size, "k")
            plt.plot(ray_x[0:ray_len + 1] / subgrid_size, ray_y[0:ray_len + 1] / subgrid_size, "kx", markersize=1)
            plt.show()
            """
            break
        last_vect_x = ray_x[ray_len] - last_x
        last_x = ray_x[ray_len]
        last_vect_y = ray_y[ray_len] - last_y
        last_y = ray_y[ray_len]
        ray_len += 1
        #if True: #ray_len > len(ray_x) - 2:
            #plt.contourf(rec_TTF, 100, cmap="flag")
            #plt.show()
            #plt.imshow(veln)
            #plt.plot(ray_x[0:ray_len] / 9, ray_y[0:ray_len] / 9)
            #plt.plot(ray_x[0:ray_len] / 9, ray_y[0:ray_len] / 9, "kx")
            #plt.plot(ray_x[ray_len - 2:ray_len] / 9, ray_y[ray_len - 2:ray_len] / 9, "r")
            #plt.plot(source[0] / 9, source[1] / 9, "rx")
            #plt.plot(receiver[0] / 9, receiver[1] / 9, "gx")
            #plt.plot(ray_x / 9, ray_y / 9, "kx")
            #plt.show()
        #plt.figure(figsize=(6, 6))
        #print(dir_index)
        #plt.imshow(veln)
        #plt.imshow(vel_map)
        #plt.gca().invert_yaxis()
        #plt.plot(ray_x[0:ray_len]/subgrid_size, ray_y[0:ray_len]/subgrid_size)
        #plt.plot(ray_x[ray_len - 1]/subgrid_size, ray_y[ray_len - 1]/subgrid_size, "bx")
        #plt.plot([source[0] / subgrid_size, receiver[0] / subgrid_size], [source[1] / subgrid_size, receiver[1] / subgrid_size], "rx")
        #plt.plot([(c_value/subgrid_size), (c_value/subgrid_size)], [0, 200])
        #plt.pause(0.05)
        #.cla()
        #plt.show()

    # Add the final point to the ray path (receiver) and remove all unused values in the arrays
    ray_x[ray_len] = receiver[0]
    ray_y[ray_len] = receiver[1]
    ray_x = ray_x[0: ray_len + 1]
    ray_y = ray_y[0: ray_len + 1]
    #if ray_len < 10:
    #    print(f"ray_len[{ray_len}]")
    #plt.imshow(veln)
    #plt.plot(ray_x/subgrid_size, ray_y/subgrid_size)
    #plt.show()
    # Integrate along the ray path to determine the travel time.
    trav_time = ray_time(ray_x, ray_y, dnx, subgrid_size, velocity_dat, veln, velpn, vel_map, stif_den)
    return ray_x, ray_y, trav_time


def slown_d_slown_stif(angle, c_22, c_23, c_33, c_44, sigma, vel_scale=1):
    """
    Function for returning the first derivative of the group velocity from material properties w.r.t group angle.

    :param angle: Effective group angle (0-180)
    :type angle: float
    :param c_22: Stifness parameter in MPa.
    :type c_22: int
    :param c_23: Stifness parameter in MPa.
    :type c_23: int
    :param c_33: Stifness parameter in MPa.
    :type c_33: int
    :param c_44: Stifness parameter in MPa.
    :type c_44: int
    :param sigma: Density in Kg/m^3
    :type sigma: int
    :param vel_scale: Velocity scaling parameter
    :type vel_scale: float
    :return: Derivative of group velocity
    :rtype: float
    """
    def group_vel(angle):
        if angle % 90 < 0.01 or angle % 90 > 90 - 0.01:
            if abs((angle % 180) - 90) < 1:
                lambda_val = c_33
            else:
                lambda_val = c_22
            return 1000 * vel_scale * math.sqrt(lambda_val / sigma)
        else:
            tan_ang = math.tan(math.radians(angle))
            A = c_22 + c_33 - 2 * c_44
            B = (c_23 + c_44) * (tan_ang - 1 / tan_ang)
            C = c_22 - c_33
            if angle < 90:
                phase_angle_rad = math.atan((-B - math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
            else:
                phase_angle_rad = math.atan((-B + math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
            lambda_val = 0.5 * (math.cos(2 * phase_angle_rad) * (c_22 - c_44) + math.sin(2 * phase_angle_rad) * (c_23 + c_44) * tan_ang + c_22 + c_44)
            return 1000 * vel_scale * math.sqrt(lambda_val / sigma) / math.cos(math.radians(angle) - phase_angle_rad)
    if angle % 90 < 0.01 or angle % 90 > 90 - 0.01:
        return 0
    elif angle % 90 < 45:
        slown1 = 1 / group_vel(angle)
        slown2 = 1 / group_vel(angle + 0.01)
        #return [slown1, (slown1 - slown2) / 0.01]
        return (slown1 - slown2) / 0.01
    else:
        slown1 = 1 / group_vel(angle)
        slown2 = 1 / group_vel(angle - 0.01)
        #return [slown1, (slown1 - slown2) / -0.01]
        return (slown1 - slown2) / -0.01

@njit(cache=True)
def group_vel(angle, c_22, c_23, c_33, c_44, sigma, vel_scale=1):
    """
        Function for the group velocity from material properties w.r.t group angle.

        :param angle: Effective group angle (0-180)
        :type angle: float
        :param c_22: Stifness parameter in MPa.
        :type c_22: int
        :param c_23: Stifness parameter in MPa.
        :type c_23: int
        :param c_33: Stifness parameter in MPa.
        :type c_33: int
        :param c_44: Stifness parameter in MPa.
        :type c_44: int
        :param sigma: Density in Kg/m^3
        :type sigma: int
        :param vel_scale: Velocity scaling parameter, default is 1.
        :type vel_scale: float
        :return: Derivative of group velocity
        :rtype: float
        """
    if angle % 90 < 0.01 or angle % 90 > 90 - 0.01:
        if abs((angle % 180) - 90) < 1:
            lambda_val = c_33
        else:
            lambda_val = c_22
        return 1000 * vel_scale * math.sqrt(lambda_val / sigma)
    else:
        tan_ang = math.tan(math.radians(angle))
        A = c_22 + c_33 - 2 * c_44
        B = (c_23 + c_44) * (tan_ang - 1 / tan_ang)
        C = c_22 - c_33
        if angle < 90:
            phase_angle_rad = math.atan((-B - math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
        else:
            phase_angle_rad = math.atan((-B + math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
        lambda_val = 0.5 * (math.cos(2 * phase_angle_rad) * (c_22 - c_44) + math.sin(2 * phase_angle_rad) * (c_23 + c_44) * tan_ang + c_22 + c_44)
        return 1000 * vel_scale * math.sqrt(lambda_val / sigma) / math.cos(math.radians(angle) - phase_angle_rad)

def parallel_TTF(queue1, queue2, nsts, btg, ntr, ttn, veln, velpn, vel_map, stif_den, velocity_dat, phase_vel, gox, goz, dnx, dnz, nnx, nnz, low_mem):
    """
    Function used in class for calculating travel time fields in parallel. All variables which remain unchanged for different sources are used to initialise the process.

    :param queue1: Queue used for receiving jobs to complete from the main process.
    :type queue1: multiprocessing Queue
    :param queue2: Queue used for returning completed jobs back to the main process.
    :type queue2: multiprocessing Queue
    :param nsts: Node status for points in the array. -1 is for unknown point, if point is still in the heap then value is the position in the tree. Should have values of -1 for start.
    :type nsts: 2D numpy array of type int
    :param btg: Positions of points in the min heap.
    :type btg: 2D numpy array of type int
    :param ntr: Number of points in binary tree. Should start with 0.
    :type ntr: int
    :param ttn: Current travel time at all points in the grid (0 for far points).
    :type ttn: 2D numpy array
    :param veln: Anisotropic orientation of all grid points.
    :type veln: 2D numpy array
    :param velpn: Material index of all grid points (0 if using stiffness tensors and density, otherwise index for column in avlist2).
    :type velpn: 2D numpy array of type int
    :param vel_map: Value used for scaling velocities at all grid points (mainly used for isotropic materials).
    :type vel_map: 2D numpy array
    :param stif_den: Material parameters with first two indices being the i,j coordinates and the third being the index of the material parameter(c_22, c_23, c_33, c_44, density). Stiffness tensors must be in MPa to avoid overflow errors.
    :type stif_den: 3D numpy array of type np.int64
    :param velocity_dat: Group velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type velocity_dat: 2D numpy array
    :param phase_vel: Phase velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type phase_vel: 2D numpy array
    :param gox: x position of the point with indices (0, 0).
    :type gox: float
    :param goz: z position of the point with indices (0, 0).
    :type goz: float
    :param dnx: Distance between points in the grid in the x direction.
    :type dnx: float
    :param dnz: Distance between points in the grid in the z direction. Must equal dnx.
    :type dnz: float
    :param nnx: Number of points in the grid in the x direction.
    :type nnx: int
    :param nnz: Number of points in the grid in the z direction.
    :type nnz: int
    :param low_mem: Parameter used when there are issues with insufficient memory. If True then saves travel time fields to the current directory as "temp_TTF_i.npy" for source index i.
    :type low_mem: bool
    :return: Does not return anything. Process should be terminated by main process when finished.
    """
    # Run forever until terminated
    while True:
        # Obtain the index of the source and the x and z position of the source. If there is no values in the queue will wait until one is available or the process is terminated.
        [i, x, z] = queue1.get()
        if low_mem == False:
            # If low_mem is False then the index of the source and the travel time field is send back to the main process (index of the source is required so the main process knows which travel time field has been calculated.
            queue2.put([i, travel(x, z, nsts, btg, ntr, ttn, veln, velpn, vel_map, stif_den, velocity_dat, phase_vel, gox, goz, dnx, dnz, nnx, nnz)])
        else:
            # If low_mem is True then the travel time field is saved to the current directory and the travel time field is returned as None (main process does not expect a travel time field).
            TTF = travel(x, z, nsts, btg, ntr, ttn, veln, velpn, vel_map, stif_den, velocity_dat, phase_vel, gox, goz, dnx, dnz, nnx, nnz)
            np.save("temp_TTF_" + str(i) + ".npy", TTF)
            queue2.put([i, None])


def parallel_TTF_finer_grid(queue1, queue2, veln, velpn, vel_map, stif_den, subgrid_size, velocity_dat, phase_vel, gox, goz, dnx, dnz, low_mem):
    """
    Function used in class for calculating travel time fields in parallel. All variables which remain unchanged for different sources are used to initialise the process.

    :param queue1: Queue used for receiving jobs to complete from the main process.
    :type queue1: multiprocessing Queue
    :param queue2: Queue used for returning completed jobs back to the main process.
    :type queue2: multiprocessing Queue
    :param veln: Anisotropic orientation of all grid points.
    :type veln: 2D numpy array
    :param velpn: Material index of all grid points (0 if using stiffness tensors and density, otherwise index for column in avlist2).
    :type velpn: 2D numpy array of type int
    :param vel_map: Value used for scaling velocities at all grid points (mainly used for isotropic materials).
    :type vel_map: 2D numpy array
    :param stif_den: Material parameters with first two indices being the i,j coordinates and the third being the index of the material parameter(c_22, c_23, c_33, c_44, density).
    :type stif_den: 3D numpy array of type np.int64
    :param subgrid_size: The size increase of the finer grid. Must be an odd integer so that points match in the original and finer grid.
    :type subgrid_size: int
    :param velocity_dat: Group velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type velocity_dat: 2D numpy array
    :param phase_vel: Phase velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type phase_vel: 2D numpy array
    :param gox: x position of the point with indices (0, 0).
    :type gox: float
    :param goz: z position of the point with indices (0, 0).
    :type goz: float
    :param dnx: Distance between points in the grid in the x direction.
    :type dnx: float
    :param dnz: Distance between points in the grid in the z direction. Must equal dnx.
    :type dnz: float
    :param low_mem: Parameter used when there are issues with insufficient memory. If True then saves travel time fields to the current directory as "temp_TTF_i.npy" for source index i.
    :type low_mem: bool
    :return: Does not return anything. Process should be terminated by main process when finished.
    """
    # Run forever until terminated
    while True:
        # Obtain the index of the source and the x and z position of the source. If there is no values in the queue will wait until one is available or the process is terminated.
        [i, x, z] = queue1.get()
        #print(f"starting source {i}\n", end="")
        if low_mem == False:
            # If low_mem is False then the index of the source and the travel time field is send back to the main process (index of the source is required so the main process knows which travel time field has been calculated.
            queue2.put([i, travel_finer_grid(x, z, veln, velpn, vel_map, stif_den, subgrid_size, velocity_dat, phase_vel, gox, goz, dnx, dnz)])
        else:
            # If low_mem is True then the travel time field is saved to the current directory and the travel time field is returned as None (main process does not expect a travel time field).
            #try:
            #    TTF = travel_finer_grid(x, z, veln, velpn, vel_map, stif_den, subgrid_size, velocity_dat, phase_vel, gox, goz, dnx, dnz)
            #    np.save("temp_TTF_" + str(i) + ".npy", TTF)
            #except:
            #    print(f"TTF with source {i} failed\n", end="")
            TTF = travel_finer_grid(x, z, veln, velpn, vel_map, stif_den, subgrid_size, velocity_dat, phase_vel, gox, goz, dnx, dnz)
            np.save("temp_TTF_" + str(i) + ".npy", TTF)
            TTF = None
            queue2.put([i, None])
        #print(f"finished source {i}\n", end="")


def parallel_TTF_rays(proc_num, queue1, queue2, trans_pairs, veln, velpn, vel_map, stif_den, subgrid_size, velocity_dat, phase_vel, gox, goz, dnx, scx, scz, new_trans_x, new_trans_z):
    """
    Function used in class for calculating travel time fields and ray paths in parallel. All variables which remain unchanged for different sources are used to initialise the process. Uses queue2 with return codes for telling main process when a travel time field is completed and for returning ray paths.

    :param proc_num: Process number.
    :type proc_num: int
    :param queue1: Queue used for receiving jobs to complete from the main process.
    :type queue1: multiprocessing Queue
    :param queue2: Queue used for telling main process a travel time field has been calculated or for sending ray paths to main process (using return codes 0 for TTF and 1 for ray).
    :type queue2: multiprocessing Queue
    :param veln: Anisotropic orientation of all grid points.
    :type veln: 2D numpy array
    :param velpn: Material index of all grid points (0 if using stiffness tensors and density, otherwise index for column in avlist2).
    :type velpn: 2D numpy array of type int
    :param vel_map: Value used for scaling velocities at all grid points (mainly used for isotropic materials).
    :type vel_map: 2D numpy array
    :param stif_den: Material parameters with first two indices being the i,j coordinates and the third being the index of the material parameter(c_22, c_23, c_33, c_44, density).
    :type stif_den: 3D numpy array of type int64
    :param subgrid_size: The size increase of the finer grid. Must be an odd integer so that points match in the original and finer grid.
    :type subgrid_size: int
    :param velocity_dat: Group velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type velocity_dat: 2D numpy array
    :param phase_vel: Phase velocity of materials at different angles (column 0 is angle i.e 0-360 and other columns are velocity for that angle).
    :type phase_vel: 2D numpy array
    :param gox: x position of the point with indices (0, 0).
    :type gox: float
    :param goz: z position of the point with indices (0, 0).
    :type goz: float
    :param dnx: Distance between points in the grid in the x direction.
    :type dnx: float
    :param scx: Array for x coordinates for transducers.
    :type scx: 1D numpy array
    :param scz: Array for z coordinates for transducers.
    :type scz: 1D numpy array
    :param new_trans_x: Array for x coordinates for transducers on finer grid using grid coordinates.
    :type new_trans_x: 1D numpy array
    :param new_trans_z: Array for z coordinates for transducers on finer grid using grid coordinates.
    :type new_trans_z: 1D numpy array
    :return: Does not return anything. Process should be terminated by main process when finished.
    """

    n_trans = trans_pairs.shape[0]
    # Run forever until terminated
    while True:
        # Obtain the index of the source and the x and z position of the source. If there is no values in the queue will wait until one is available or the process is terminated.
        j = queue1.get()
        receiver = np.array([new_trans_x[j], new_trans_z[j]])
        TTF = travel_finer_grid(scx[j], scz[j], veln, velpn, vel_map, stif_den, subgrid_size, velocity_dat, phase_vel, gox, goz, dnx, dnx)
        #np.save(f"TTF_{j}.npy", TTF)
        queue2.put([0, j])

        for i in range(n_trans):
            if trans_pairs[i, j] == 1:
                source = np.array([new_trans_x[i], new_trans_z[i]])
                ray_x, ray_y, time = find_ray(dnx, velocity_dat, source, receiver, TTF, veln, velpn, vel_map, stif_den, subgrid_size)
                ray_x = ray_x / subgrid_size
                ray_y = ray_y / subgrid_size
                #np.save(f"./target_rays/ray_x_{i}_{j}.npy", ray_x)
                #np.save(f"./target_rays/ray_y_{i}_{j}.npy", ray_y)
                queue2.put([1, [i, j, ray_x, ray_y, time]])


@njit(cache=True)
def min_max_vel(veln, velpn, vel_map, stif_den, group_vel_table):
    """
    Function for determining the min / max velocity in a model to check if model is input correctly.

    :param veln: Anisotropic material orientations at grid points. Set as array of zero if using isotropic materials.
    :type veln: 2D numpy array
    :param velpn: Material index of each grid point (0 if using stiffness tensors and density, otherwise index for column in velocity table).
    :type velpn: 2D numpy array of type int
    :param vel_map: Values used to scaling velocity for each grid point. Use array of ones for anisotropic materials, or array of velocities for isotropic materials (using material with velocity curve with velocity of 1) . If unused then an array of ones is used.
    :type vel_map: 2D numpy array
    :param stif_den: Stiffness tensors at each grid point. First two indices are the position of the grid point and 3rd index is for the materials parameters, 0 - c_22, 1 - c_23, 2 - c_33, 3 - c_44, 4 - density. Array must use 64 bit integers with stiffness tensors in MPa and density in Kg/m^3. To use these values the material index of grid points should be 0. If a point is not using stiffness tensors and density the values are not used. If parameter not used, velocity curves will be used instead(don't set material index to 0).
    :type stif_den: 3D numpy array of type np.int64
    :param group_vel_table: Group velocity curves with first column giving the angle (0-360 with increments of 1 degree) and other columns group velocities for each material(column is material indecies). If parameter is unused then defalts to velocity curve for isotropic material with velocity of 1(use vel_map to scale curve to set velocities at each point)
    :type group_vel_table: 2D numpy array
    :return: min_vel, max_vel. The minimum and maximum velocity in a model
    :rtype: float, float
    """

    group_vel_max = np.zeros(group_vel_table.shape[1])
    group_vel_min = np.copy(group_vel_max)
    for i in range(len(group_vel_min)):
        group_vel_min[i] = np.min(group_vel_table[:, i])
        group_vel_max[i] = np.max(group_vel_table[:, i])

    if velpn[0, 0] == 0:
        c_22 = stif_den[0, 0, 0]
        c_23 = stif_den[0, 0, 1]
        c_33 = stif_den[0, 0, 2]
        c_44 = stif_den[0, 0, 3]
        density = stif_den[0, 0, 4]
        min_vel = group_vel(0, c_22, c_23, c_33, c_44, density, vel_map[0, 0])
        max_vel = min_vel
    else:
        min_vel = vel_map[0, 0] * group_vel_table[0, velpn[0, 0]]
        max_vel = min_vel
    for i in range(veln.shape[0]):
        for j in range(veln.shape[1]):
            if velpn[0, 0] == 0:
                c_22 = stif_den[i, j, 0]
                c_23 = stif_den[i, j, 1]
                c_33 = stif_den[i, j, 2]
                c_44 = stif_den[i, j, 3]
                density = stif_den[i, j, 4]
                for angle in [0, 45, 90, 135]:
                    vel = group_vel(angle, c_22, c_23, c_33, c_44, density, vel_map[i, j])
                    min_vel = min(min_vel, vel)
                    max_vel = max(max_vel, vel)
            else:
                min_vel = min(min_vel, vel_map[i, j] * group_vel_min[velpn[i, j]])
                max_vel = max(max_vel, vel_map[i, j] * group_vel_max[velpn[i, j]])
    return min_vel, max_vel

class ALI_FMM:
    """
    Class for calculating travel time fields and performing ray tracing through the travel time fields.
    """
    def __init__(self, veln, velpn, vel_map, scx, scz, group_vel=None, phase_vel=None, stif_den=None, dnx=1e-3):
        """
        Function to initialise class

        :param veln: Anisotropic material orientations at grid points. Set as array of zeros if using isotropic materials.
        :type veln: 2D numpy array
        :param velpn: Material index of each grid point (0 if using stiffness tensors and density, otherwise index for column in velocity table).
        :type velpn: 2D numpy array of type int
        :raises TypeError: Creates error if velpn is not array of integer
        :param vel_map: Velocity scaling parameters for each grid point. Use array of ones for anisotropic materials, or array of velocities for isotropic materials.
        :type vel_map: 2D numpy array
        :param scx: numpy array of x coordinates for sources/recievers.
        :type scx: 1D numpy array
        :param scz: numpy array of z coordinates for sources/recievers.
        :type scz: 1D numpy array
        :param group_vel: Group velocity curves with first column giving the angle (0-360 with increments of 1 degree) and other columns group velocities for each material(column is material indecies). If parameter is unused then defalts to velocity curve for isotropic material with velocity of 1(use vel_map to scale curve to set velocities at each point)
        :type group_vel: 2D numpy array
        :param phase_vel: Phase velocity curves with first column giving the angle (0-360 with increments of 1 degree) and other columns phase velocities for each material(column is material indecies). If parameter is unused then defalts to velocity curve for isotropic material with velocity of 1(use vel_map to scale curve to set velocities at each point)
        :type phase_vel: 2D numpy array
        :param stif_den: Stifness tensors at each grid point. First two indecies are the position of the grid point and 3rd index is for the materials parameters, 0 - c_22, 1 - c_23, 2 - c_33, 3 - c_44, 4 - density. Array must use 64 bit integers with stiffness tensors in MPa and density in Kg/m^3. To use these values the material index of grid points should be 0. If a point is not using stiffness tensors and density the values are not used. If parameter not used, velocity curves will be used instead(don't set material index to 0).
        :type stif_den: 3D numpy array of type np.int64
        :raises TypeError: Creates error if stiffness tensors are not the correct type (np.int64). Does not create error if stifness tensors are not being used (None used)/
        :param dnx: Spacing between each point in metres, defalts as 1e-3.
        :type dnx: float
        """
        # Initialise all the parameters to the class
        self.stif_den = stif_den  # All stifness tensors in MPa not Pa, due to 32 bit numbers
        if type(stif_den) != type(None):
            if type(stif_den[0, 0, 0]) != np.int64:
                raise TypeError("Stifness tensors and density array must have the type np.int64. 32bit integers will not work correctly.")
            elif stif_den[0, 0, 0] > 1e9:
                print("Warning: Stifness tensors must be in MPa, due to 64 bit integer limitations when solving the christoffel equation")
        if type(group_vel) == type(None):
            self.velocity_dat = 1 * np.ones((361, 2))
            self.velocity_dat[:, 0] = np.arange(0, 361)
            self.phase_vel = np.copy(self.velocity_dat)
        else:
            self.velocity_dat = group_vel
            self.phase_vel = phase_vel
        self.veln = veln
        self.velpn = velpn
        try:
            if np.issubdtype(velpn[0, 0], np.integer) == False:
                raise TypeError("velpn must be a numpy array of integers")
        except:
            raise TypeError("velpn must be a numpy array of integers")
        self.vel_map = vel_map
        self.dnx = dnx
        self.dnz = dnx
        self.nnx = veln.shape[1]
        self.nnz = veln.shape[0]
        self.ttn = np.zeros(veln.shape)
        self.scx = scx
        self.scz = scz
        self.gox = 0
        self.goz = 0
        self.isx = np.zeros(len(scx))
        self.isz = np.zeros(len(scx))
        for i in range(len(scx)):
            self.isx[i] = round((scx[i] - self.gox) / self.dnx)
            self.isz[i] = round((scz[i] - self.goz) / self.dnz)
        self.ntr = 0
        self.nsrc = len(scx)
        #self.travel_time_field = np.zeros((self.nsrc, self.nnz, self.nnx))

        # Allocate memory for node status and binary trees
        snb = 0.5
        self.nsts = np.zeros((self.nnx, self.nnz), dtype=int)
        self.maxbt = round(snb * self.nnx * self.nnz)
        self.btg = np.zeros((self.maxbt, 2), dtype=int)

        # Initialise parameters for storing ray paths in a 2D array and the number of points in the ray path (since the ray paths have different lengths a function is used for obtaining the ray paths).
        self.ray_paths_x = None
        self.ray_paths_y = None
        self.ray_len = None


    def update(self, veln, velpn, vel_map=None, stif_den=None, subgrid_size=1, sources=None):
        """
        Function for computing travel time fields from material properties.

        :param veln: Anisotropic material orientations at grid points. Set as array of zero if using isotropic materials.
        :type veln: 2D numpy array
        :param velpn: Material index of each grid point (0 if using stiffness tensors and density, otherwise index for column in velocity table).
        :type velpn: 2D numpy array of type int
        :param vel_map: Values used to scaling velocity for each grid point. Use array of ones for anisotropic materials, or array of velocities for isotropic materials (using material with velocity curve with velocity of 1) . If unused then an array of ones is used.
        :type vel_map: 2D numpy array
        :param stif_den: Stiffness tensors at each grid point. First two indices are the position of the grid point and 3rd index is for the materials parameters, 0 - c_22, 1 - c_23, 2 - c_33, 3 - c_44, 4 - density. Array must use 64 bit integers with stiffness tensors in MPa and density in Kg/m^3. To use these values the material index of grid points should be 0. If a point is not using stiffness tensors and density the values are not used. If parameter not used, velocity curves will be used instead(don't set material index to 0).
        :type stif_den: 3D numpy array of type np.int64
        :param subgrid_size: Parameter for computing travel time field on finer grid (must be a odd number), multiply indices by subgrid_size to move between indices on original grid. Default value is set to 1 i.e same as original grid.
        :type subgrid_size: int
        :param sources: Array of 0's and 1's for selecting which sources should be used for calculating travel time fields. If unused all sources will be used.
        :type sources: 1D numpy array
        :return: Travel time fields for all sources. If sources parameter used then sources not used return array of zeros.
        :rtype: 3D numpy array
        """
        # If stifness tensors and density's are not used then array of zeros are used (will never be used).
        if type(stif_den) == type(None):
            self.stif_den = np.zeros((veln.shape[0], veln.shape[1], 5))
        else:
            self.stif_den = stif_den
        self.veln = veln
        self.velpn = velpn
        # If no velocity scaling is required then scaling defaults to 1 i.e. does not scale.
        if type(vel_map) == type(None):
            self.vel_map = np.ones(veln.shape)
        else:
            self.vel_map = vel_map
        # If no value is input to sources then all travel time fields are calculated.
        if type(sources) == type(None):
            sources = np.ones(len(self.scx))
        if subgrid_size == 1:
            # Initialise array for storing travel time fields
            travel_time_field = np.zeros((self.nsrc, veln.shape[0], veln.shape[1]))
            for i in tqdm(range(self.nsrc), disable=tqdm_disable, colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]"):  # for i=1:nsrc  ", ncols=100"
                if sources[i] == 1:
                    #print(i + 1)
                    # Array index of transmitter
                    x = float(self.scx[i])
                    z = float(self.scz[i])
                    # Call a subroutine that works out the first - arrival traveltime field.
                    ttn = np.zeros(self.veln.shape)
                    travel_time_field[i, :, :] = travel(x, z, self.nsts, self.btg, self.ntr, self.ttn, self.veln, self.velpn, self.vel_map, self.stif_den, self.velocity_dat, self.phase_vel, self.gox, self.goz, self.dnx, self.dnz, self.nnx, self.nnz)
                else:
                    travel_time_field[i, :, :] = np.zeros(veln.shape)
            return travel_time_field
        else:
            first_TTF = True
            for i in tqdm(range(self.nsrc), disable=tqdm_disable, ncols=100, colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]"):  # for i=1:nsrc
                if sources[i] == 1:
                    #print(i + 1)
                    # Array index of transmitter
                    x = float(self.scx[i])
                    z = float(self.scz[i])
                    # Call a subroutine that works out the first - arrival traveltime field.
                    if first_TTF == True:
                        # First travel time field determines size of array used for storing arrays.
                        temp_ttf = travel_finer_grid(x, z, self.veln, self.velpn, self.vel_map, self.stif_den, subgrid_size, self.velocity_dat, self.phase_vel, self.gox, self.goz, self.dnx, self.dnz)
                        travel_time_field = np.zeros((self.nsrc, temp_ttf.shape[0], temp_ttf.shape[1]))
                        travel_time_field[i, :, :] = temp_ttf
                        first_TTF = False
                    else:
                        travel_time_field[i, :, :] = travel_finer_grid(x, z, self.veln, self.velpn, self.vel_map, self.stif_den, subgrid_size, self.velocity_dat, self.phase_vel, self.gox, self.goz, self.dnx, self.dnz)
            return travel_time_field

    def update_parallel(self, veln, velpn, vel_map=None, stif_den=None, subgrid_size=1, sources=None, n_threads=2, low_mem=False):
        """
        Function for computing travel time fields from material properties in parallel. Parallelization is only possible for different travel time fields. Since there is a computational cost to setting up parallelization, this is not recommended for small grids as will likely take longer.

        :param veln: Anisotropic material orientations at grid points. Set as array of zero if using isotropic materials.
        :type veln: 2D numpy array
        :param velpn: Material index of each grid point (0 if using stiffness tensors and density, otherwise index for column in velocity table).
        :type velpn: 2D numpy array of type int
        :param vel_map: Values used to scaling velocity for each grid point. Use array of ones for anisotropic materials, or array of velocities for isotropic materials (using material with velocity curve with velocity of 1) . If unused then an array of ones is used.
        :type vel_map: 2D numpy array
        :param stif_den: Stiffness tensors at each grid point. First two indices are the position of the grid point and 3rd index is for the materials parameters, 0 - c_22, 1 - c_23, 2 - c_33, 3 - c_44, 4 - density. Array must use 64 bit integers with stiffness tensors in MPa and density in Kg/m^3. To use these values the material index of grid points should be 0. If a point is not using stiffness tensors and density the values are not used. If parameter not used, velocity curves will be used instead(don't set material index to 0).
        :type stif_den: 3D numpy array of type np.int64
        :param subgrid_size: Parameter for computing travel time field on finer grid (must be a odd number), multiply indices by subgrid_size to move between indices on original grid. Default value is set to 1 i.e same as original grid.
        :type subgrid_size: int
        :param sources: Array of 0's and 1's for selecting which sources should be used for calculating travel time fields. If unused all sources will be used.
        :type sources: 1D numpy array
        :param n_threads: Number of threads to be used. If n_threads = 1 then use update instead of update parallel.
        :type n_threads: int
        :param low_mem: whether to use less memory or not. True saves travel time fields to directory as "temp_TTF_i.npy"(load using np.load) for TTF i and False saves them in memory.
        :type low_mem: bool
        :return: Travel time fields for all sources. If sources parameter used then sources not used will return array of zeros. If low_mem=True then returns None.
        :rtype: 3D numpy array/None
        """
        # If stifness tensors and density's are not used then array of zeros are used (will never be used).
        if type(stif_den) == type(None):
            self.stif_den = np.zeros((veln.shape[0], veln.shape[1], 5))
        else:
            self.stif_den = stif_den
        self.veln = veln
        self.velpn = velpn
        # If no velocity scaling is required then scaling defalts to 1 i.e. does not scale.
        if type(vel_map) == type(None):
            self.vel_map = np.ones(veln.shape)
        else:
            self.vel_map = vel_map
        # If sources parameter is  not used then all sources are used.
        if type(sources) == type(None):
            sources = np.ones(len(self.scx), dtype=int)
        # Initialise queue for sending data between processes.
        queue1 = multiprocessing.Queue()
        queue2 = multiprocessing.Queue()
        for i in range(self.nsrc):  # for i=1:nsrc
            if sources[i] == 1:
                # Add all required sources to the queue so processes can retrieve jobs to complete.
                x = float(self.scx[i])
                z = float(self.scz[i])
                queue1.put([i, x, z])
        if subgrid_size == 1:
            ttn = np.zeros(self.veln.shape)

            # Initialise processes
            processes = []
            for i in range(n_threads):
                process = multiprocessing.Process(target=parallel_TTF, args=(queue1, queue2, self.nsts, self.btg, self.ntr, self.ttn, self.veln, self.velpn, self.vel_map, self.stif_den, self.velocity_dat, self.phase_vel, self.gox, self.goz, self.dnx, self.dnz, self.nnx, self.nnz, low_mem))
                processes.append(process)
            # Start all processes running.
            for process in processes:
                process.start()
            if low_mem == False:
                # Initialise array for storing travel time fields
                travel_time_field = np.zeros((self.nsrc, veln.shape[0], veln.shape[1]))
                # Retrieve all travel time fields from processes and add them to array of travel time fields.
                for i in tqdm(range(int(np.sum(sources))), disable=tqdm_disable, ncols=100, desc="Finished TTF's", colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]"):
                    # Waits until a travel time field is recieved
                    index, TTF = queue2.get()
                    travel_time_field[index, :, :] = TTF
            else:
                travel_time_field = None
                # Waits until all travel time fields have been calculated
                for i in tqdm(range(int(np.sum(sources))), disable=tqdm_disable, ncols=100, desc="Finished TTF's", colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]"):
                    queue2.get()
            # Tells all processes to terminate
            for process in processes:
                process.terminate()
            # Will wait until all processes have finished terminating
            for process in processes:
                process.join()
            # Will remove all processes so memory is released back to os.
            for process in processes:
                process.close()
            return travel_time_field
        else:
            # Uses a finer grid

            # Set up processes
            processes = []
            for i in range(n_threads):
                process = multiprocessing.Process(target=parallel_TTF_finer_grid, args=(queue1, queue2, self.veln, self.velpn, self.vel_map, self.stif_den, subgrid_size, self.velocity_dat, self.phase_vel, self.gox, self.goz, self.dnx, self.dnz, low_mem))
                processes.append(process)
                process.start()
            if low_mem == False:
                # Waits until the first travel time field is retrieved and used to set size of travel time fields.
                index, TTF = queue2.get()
                travel_time_field = np.zeros((self.nsrc, TTF.shape[0], TTF.shape[1]))
                travel_time_field[index, :, :] = TTF
                # As travel time fields are retrieved they are stored into an array of travel time fields.
                for i in tqdm(range(int(np.sum(sources)) - 1), disable=tqdm_disable, ncols=100, desc="Finished TTF's", colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]"):
                    index, TTF = queue2.get()
                    travel_time_field[index, :, :] = TTF
            else:
                travel_time_field = None
                # Waits until all travel time fields are completed.
                for i in tqdm(range(int(np.sum(sources))), disable=tqdm_disable, ncols=100, desc="Finished TTF's", colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]"):
                    queue2.get()
            # Tells all processes to terminate
            for process in processes:
                process.terminate()
            # Will wait until all processes have finished terminating
            for process in processes:
                process.join()
            # Will remove all processes so memory is released back to os.
            for process in processes:
                process.close()
            return travel_time_field

    def update_i(self, source_i, veln, velpn, vel_map, stif_den=None, subgrid_size=1):
        """
        Function for computing a single travel time field.

        :param source_i: Index of the source.
        :type source_i: int
        :param veln: Anisotropic material orientations at grid points. Set as array of zeros if using isotropic materials.
        :type veln: 2D numpy array
        :param velpn: Material index of each grid point (0 if using stiffness tensors and density, otherwise index for column in velocity table).
        :type velpn: 2D numpy array of type int
        :param vel_map: Values used for scaling velocities at each grid point. Use array of ones for anisotropic materials, or array of velocities for isotropic materials.
        :type vel_map: 2D numpy array
        :param stif_den: Stiffness tensors at each grid point. First two indices are the position of the grid point and 3rd index is for the materials parameters, 0 - c_22, 1 - c_23, 2 - c_33, 3 - c_44, 4 - density. Array must use 64 bit integers with stiffness tensors in MPa and density in Kg/m^3. To use these values the material index of grid points should be 0. If a point is not using stiffness tensors and density the values are not used. If parameter not used, velocity curves will be used instead(don't set material index to 0).
        :type stif_den: 3D numpy array of type np.int64
        :param subgrid_size: Parameter for computing travel time field on finer grid (must be a odd number), multiply indies by subgrid_size to move between indices on original grid. Default value is set to 1 i.e same as original grid.
        :type subgrid_size: int
        :return: Travel time field for the selected source.
        :rtype: 2D numpy array
        """
        # If no velocity scaling is required/
        if type(vel_map) == type(None):
            vel_map = np.ones(veln.shape)
        # If stiffness tensors and density is not being used.
        if type(stif_den) == type(None):
            stif_den = np.zeros((veln.shape[0], veln.shape[1], 5))
        # Calculate travel time field
        if subgrid_size == 1:
            x = float(self.scx[source_i])
            z = float(self.scz[source_i])
            # Call a subroutine that works out the first - arrival traveltime field.
            ttn = np.zeros(veln.shape)
            return travel(x, z, self.nsts, self.btg, self.ntr, self.ttn, veln, velpn, vel_map, stif_den, self.velocity_dat, self.phase_vel, self.gox, self.goz, self.dnx, self.dnz, self.nnx, self.nnz)
        else:
            x = float(self.scx[source_i])
            z = float(self.scz[source_i])
            return travel_finer_grid(x, z, veln, velpn, vel_map, stif_den, subgrid_size, self.velocity_dat, self.phase_vel, self.gox, self.goz, self.dnx, self.dnz)

    def plot_phase(self, material_index=1):
        """
        Plots the phase velocity curve for a material defined by table of velocities.

        :param material_index: Index of the material being plotted, default value is 1.
        :type material_index: int
        :return: Does not return anything.
        """
        plt.polar(math.pi / 180 * self.velocity_dat[:, 0], self.phase_vel[:, material_index])
        plt.show()

    def plot_group(self, material_index=1):
        """
        Plots the group velocity curve for a material defined by table of velocities.

        :param material_index: Index of the material being plotted, default value is 1.
        :type material_index: int
        :return: Does not return anything.
        """
        plt.polar(math.pi / 180 * self.velocity_dat[:, 0], self.velocity_dat[:, material_index])
        plt.show()

    def generate_group_vel(self, c_22, c_23, c_33, c_44, density, plot=True):
        """
        Generates the group velocity curve, given a materials stiffness tensors and density.

        :param c_22: Stiffness Tensor in Pa.
        :type c_22: int
        :param c_23: Stiffness Tensor in Pa.
        :type c_23: int
        :param c_33: Stiffness Tensor in Pa.
        :type c_33: int
        :param c_44: Stiffness Tensor in Pa.
        :type c_44: int
        :param density: Density of the material in Kg/m^3.
        :type density: int
        :param plot: True or False value for whether to plot curve or not. Default is True.
        :type plot: bool
        :return: Array of group velocities from 0 to 360 degrees with increments of 1 degree.
        :rtype: 2D numpy array
        """
        group_vel = np.zeros(361)
        for angle in range(361):
            if angle < 180:
                # Solves christoffel equation
                if angle % 90 == 0:
                    if angle % 180 == 90:
                        lambda_val = c_33
                    else:
                        lambda_val = c_22
                    velocity = math.sqrt(lambda_val / density)
                else:
                    tan_ang = math.tan(math.radians(angle))
                    A = c_22 + c_33 - 2 * c_44
                    B = (c_23 + c_44) * (tan_ang - 1 / tan_ang)
                    C = c_22 - c_33
                    if angle < 90:
                        phase_angle_rad = math.atan((-B - math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
                    else:
                        phase_angle_rad = math.atan((-B + math.sqrt(B ** 2 + A ** 2 - C ** 2)) / (C - A)) % math.pi
                    lambda_val = 0.5 * (math.cos(2 * phase_angle_rad) * (c_22 - c_44) + math.sin(2 * phase_angle_rad) * (c_23 + c_44) * tan_ang + c_22 + c_44)
                    velocity = math.sqrt(lambda_val / density) / math.cos(math.radians(angle) - phase_angle_rad)
                group_vel[angle] = velocity
            else:
                group_vel[angle] = group_vel[angle - 180]
        # Plots group velocity curve if required
        if plot == True:
            plt.polar(math.pi / 180 * np.arange(0, 361), group_vel)
            plt.title("Group Velocity")
            plt.show()
        return group_vel

    def generate_phase_vel(self, c_22, c_23, c_33, c_44, density, plot=True):
        """
        Generates the phase velocity curve, given a materials stiffness tensors and density.

        :param c_22: Stiffness Tensor in Pa.
        :type c_22: int
        :param c_23: Stiffness Tensor in Pa.
        :type c_23: int
        :param c_33: Stiffness Tensor in Pa.
        :type c_33: int
        :param c_44: Stiffness Tensor in Pa.
        :type c_44: int
        :param density: Density of the material in Kg/m^3.
        :type density: int
        :param plot: True or False value for whether to plot curve or not. Default is True.
        :type plot: bool
        :return: Array of phase velocities from 0 to 360 degrees with increments of 1 degree.
        :rtype: 2D numpy array
        """
        phase_vel = np.zeros(361)
        for angle in range(361):
            if angle < 180:
                # Solves christoffel equation
                if angle % 90 == 0:
                    if angle % 180 == 90:
                        lambda_val = c_33
                    else:
                        lambda_val = c_22
                    velocity = math.sqrt(lambda_val / density)
                else:
                    cos_ang = math.cos(math.radians(angle))
                    sin_ang = math.sin(math.radians(angle))
                    A = cos_ang ** 2 * c_22 + sin_ang ** 2 * c_44
                    B = cos_ang * sin_ang * (c_23 + c_44)
                    C = cos_ang ** 2 * c_44 + sin_ang ** 2 * c_33
                    velocity = math.sqrt((A + C + math.sqrt((A - C) ** 2 + 4 * B ** 2)) / (2 * density))
                phase_vel[angle] = velocity
            else:
                phase_vel[angle] = phase_vel[angle - 180]
        # Plots phase velocity curve is required.
        if plot == True:
            plt.polar(math.pi / 180 * np.arange(0, 361), phase_vel)
            plt.title("Phase Velocity")
            plt.show()
        return phase_vel

    def add_materials(self, materials, keep_materials=False):
        """
        Function for adding materials using stiffness tensors and density of materials. Prints the material indices of materials if existing materials are kept.

        :param materials: Material properties(stiffness tensors and density) for materials. Either array for single material or 2D array for multiple materials.
        :type materials: 1D array/2D array
        :param keep_materials: Whether the current materials in the class are kept when adding materials or deleted.
        :type keep_materials: bool
        :return: Nothing is returned. Materials are added into class.
        """
        if keep_materials == True:
            if materials.ndim == 1:
                group_vel_data = np.zeros((361, self.velocity_dat.shape[1] + 1))
                group_vel_data[:, 0:self.velocity_dat.shape[1]] = self.velocity_dat
                group_vel_data[:, group_vel_data.shape[1] - 1] = self.generate_group_vel(materials[0], materials[1], materials[2], materials[3], materials[4], False)
                phase_vel_data = np.zeros((361, self.phase_vel.shape[1] + 1))
                phase_vel_data[:, 0:self.velocity_dat.shape[1]] = self.phase_vel
                phase_vel_data[:, group_vel_data.shape[1] - 1] = self.generate_phase_vel(materials[0], materials[1], materials[2], materials[3], materials[4], False)
                print("material id of new material is " + str(self.velocity_dat.shape[1]))
            else:
                group_vel_data = np.zeros((361, self.velocity_dat.shape[1] + materials.shape[1]))
                group_vel_data[:, 0:self.velocity_dat.shape[1]] = self.velocity_dat
                phase_vel_data = np.zeros((361, self.velocity_dat.shape[1] + materials.shape[1]))
                phase_vel_data[:, 0:self.velocity_dat.shape[1]] = self.phase_vel
                for i in range(materials.shape[0]):
                    index = i + self.velocity_dat.shape[1]
                    group_vel_data[:, index] = self.generate_group_vel(materials[i, 0], materials[i, 1], materials[i, 2], materials[i, 3], materials[i, 4], False)
                    phase_vel_data[:, index] = self.generate_phase_vel(materials[i, 0], materials[i, 1], materials[i, 2], materials[i, 3], materials[i, 4], False)
                print("material id's of new materials are " + str(self.velocity_dat.shape[1]) + " - " + str(self.velocity_dat.shape[1] + materials.shape[0] - 1))
        else:
            if materials.ndim == 1:
                group_vel_data = np.zeros((361, 2))
                phase_vel_data = np.zeros((361, 2))
            else:
                group_vel_data = np.zeros((361, materials.shape[1] + 1))
                phase_vel_data = np.zeros((361, materials.shape[1] + 1))
            group_vel_data[:, 0] = np.arange(0, 361)
            phase_vel_data[:, 0] = np.arange(0, 361)
            if materials.ndim == 1:
                group_vel_data[:, 1] = self.generate_group_vel(materials[0], materials[1], materials[2], materials[3], materials[4], False)
                phase_vel_data[:, 1] = self.generate_phase_vel(materials[0], materials[1], materials[2], materials[3], materials[4], False)
            else:
                for i in range(materials.shape[1]):
                    index = i + 1
                    group_vel_data[:, index] = self.generate_group_vel(materials[i, 0], materials[i, 1], materials[i, 2], materials[i, 3], materials[i, 4], False)
                    phase_vel_data[:, index] = self.generate_phase_vel(materials[i, 0], materials[i, 1], materials[i, 2], materials[i, 3], materials[i, 4], False)
        # Saves group and phase velocities into the class.
        self.velocity_dat = group_vel_data
        self.phase_vel = phase_vel_data

    def find_all_TTF_rays(self, veln, velpn, vel_map=None, subgrid_size=9, trans_pairs=None, stif_den=None, save_rays=True):
        """
        Computes travel time fields for required receivers and performs ray tracing between source and receiver pairs. Returns travel times, however ray paths can be obtained using the function ray_path.

        :param veln: Anisotropic material orientations at grid points. Set as array of zeros if using isotropic materials.
        :type veln: 2D numpy array
        :param velpn: Material index of each grid point (0 if using stiffness tensors and density, otherwise index for column in velocity table).
        :type velpn: 2D numpy array of type int
        :param vel_map: Velocity scaling parameters for each grid point. Use array of ones for anisotropic materials, or array of velocities for isotropic materials. If unused then an array of ones is used.
        :type vel_map: 2D numpy array
        :param subgrid_size: Parameter for computing travel time field on finer grid (must be a odd number), multiply indices by subgrid_size to move between indices on original grid. Default value is set to 1 i.e same as original grid.
        :type subgrid_size: int
        :param trans_pairs: Transducer pairs where ray tracing is being performed (0 - No ray, 1 - compute ray). If parameter not used then all rays calculated(only one ray calculated per transducer pair i.e transducer is either source or receiver).
        :type trans_pairs: 2D numpy array
        :param stif_den: Stiffness tensors at each grid point. First two indices are the position of the grid point and 3rd index is for the materials parameters, 0 - c_22, 1 - c_23, 2 - c_33, 3 - c_44, 4 - density. Array must use 64 bit integers with stiffness tensors in MPa and density in Kg/m^3. To use these values the material index of grid points should be 0. If a point is not using stiffness tensors and density the values are not used. If parameter not used, velocity curves will be used instead(don't set material index to 0).
        :type stif_den: 3D numpy array of type int64
        :return: Travel times along the ray paths. When the path is not calculated the value is 0.
        :rtype: 2D numpy array
        """

        # If no velocity scaling is required.
        if type(vel_map) == type(None):
            vel_map = np.ones(veln.shape)
        # If stiffness tensors and density's are not used
        if type(stif_den) == type(None):
            stif_den = np.zeros((veln.shape[0], veln.shape[1], 5), dtype=np.int64)
        n_trans = len(self.isx)
        # Sets up arrays for storing ray paths.
        if save_rays:
            self.ray_paths_x = np.zeros((n_trans, n_trans, 5 * (veln.shape[0] + veln.shape[1])))
            self.ray_paths_y = np.copy(self.ray_paths_x)
            self.ray_len = np.zeros((n_trans, n_trans), dtype=int)

        if type(trans_pairs) == type(None):
            # If the ray paths that are required are not included then all combinations of transducers are used (each pair only uses one ray path i.e. transducer in each pair is either source or reciever.
            trans_pairs = np.zeros((n_trans, n_trans))
            for i in range(n_trans):
                for j in range(n_trans):
                    if i < j:
                        trans_pairs[i, j] = 1
        # Determine which travel time fields are required (if not required then travel time field is not calculated).
        rec_trans = np.zeros(n_trans)
        for j in range(n_trans):
            if sum(trans_pairs[:, j]) > 0:
                rec_trans[j] = 1
        # Calculate travel time fields.
        #TTFs = self.update(veln, velpn, vel_map, stif_den, subgrid_size, rec_trans)
        #print("Finished TTF's")
        #np.save("temp_TTFs.npy", TTFs)
        #TTFs = np.load("temp_TTFs.npy")
        #for i in range(n_trans):
        #    plt.contourf(TTFs[i, :, :], 20)
        #    plt.show()
        #plt.contour(TTFs[58, :, :], 100)
        #plt.show()


        # Calculate position of transducers on finer grid
        new_trans_x = subgrid_size * self.isx
        new_trans_y = subgrid_size * self.isz
        #print(self.isx, self.isz)
        #print(new_trans_x, new_trans_y)

        # Create array for storing travel times of ray paths.
        if type(trans_pairs) == type(None):
            trans_pairs = np.ones((n_trans, n_trans))
        times = np.zeros((n_trans, n_trans))

        n_rays = int(np.sum(trans_pairs))
        #pbar1 = tqdm(total=int(np.sum(rec_trans)), disable=tqdm_disable, ncols=100, desc="Finished TTF's", colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]")
        #pbar2 = tqdm(total=n_rays, disable=tqdm_disable, ncols=100, desc="Finished ray paths", colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]")  # , leave=False)

        pbar1 = tqdm(total=int(np.sum(rec_trans)), disable=tqdm_disable, desc="Finished TTF's", colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]")
        pbar2 = tqdm(total=n_rays, disable=tqdm_disable, desc="Finished ray paths", colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]")  # , leave=False)

        for j in range(n_trans):
            if np.sum(trans_pairs[:, j]) > 0:
                #print(f"Starting TTF [{j}]")
                #tqdm.write(f"Starting TTF [{j}]")
                TTF_j = self.update_i(j, veln, velpn, vel_map, stif_den, subgrid_size)
                #print(f"Finished TTF [{j}]")
                #tqdm.write(f"Finished TTF [{j}]")
                pbar1.update(1)
                for i in range(n_trans): #for j in range(i + 1, len(self.isx)):
                    if i != j:
                        if trans_pairs[i, j] == 1:
                            # If the ray path needs to be calculated.
                            source = np.array([new_trans_x[i], new_trans_y[i]])
                            receiver = np.array([new_trans_x[j], new_trans_y[j]])
                            #print(f"Starting Ray [{i},{j}]")
                            #tqdm.write(f"Starting Ray [{i},{j}]")
                            ray_x, ray_y, times[i, j] = find_ray(self.dnx, self.velocity_dat, source, receiver, TTF_j, veln, velpn, vel_map, stif_den, subgrid_size)
                            #print(f"Finished Ray [{i},{j}]")
                            #tqdm.write(f"Finished Ray [{i},{j}]")
                            pbar2.update(1)

                            # Bring ray paths back from finer grid to the original grid.
                            ray_x = ray_x / subgrid_size
                            ray_y = ray_y / subgrid_size

                            # Store ray path into class to be retrieved using class function ray_path
                            if save_rays:
                                ray_len = len(ray_x)
                                self.ray_paths_x[i, j, 0:ray_len] = ray_x
                                self.ray_paths_y[i, j, 0:ray_len] = ray_y
                                self.ray_len[i, j] = ray_len
        return times

    '''
    def find_all_TTF_rays_parallel(self, veln, velpn, vel_map=None, subgrid_size=9, trans_pairs=None, stif_den=None, n_threads_TTF=2, n_threads_rays=2, low_mem=False, save_rays=True):
        """
        Computes travel time fields for required receivers and performs ray tracing between source and receiver pairs using parallelisation. Returns travel times, however ray paths can be obtained using the function ray_path.

        :param veln: 2D numpy array of Anisotropic material orientations at grid points. Set as an array of zeros if using isotropic materials.
        :param velpn: 2D numpy array of type int for the material index of each grid point (0 if using stiffness tensors and density, otherwise index for column in velocity table).
        :param vel_map: 2D numpy array of scaling parameters for each grid point. Use array of ones for anisotropic materials, or array of velocities for isotropic materials. If unused then a array of ones is used.
        :param subgrid_size: Parameter for computing travel time field on finer grid (must be a odd number), multiply indices by subgrid_size to move between indices on original grid. Default value is set to 1 i.e same as original grid.
        :param trans_pairs: 2D numpy array of the transducer pairs where ray tracing is being performed (0 - No ray, 1 - compute ray). If parameter not used then all rays calculated(only one ray calculated per transducer pair i.e transducer is either source or receiver). Leading diagonal must have zeros.
        :param stif_den: 3D numpy array of type int64 for stiffness tensors at each grid point. First two indices are the position of the grid point and 3rd index is for the materials parameters, 0 - c_22, 1 - c_23, 2 - c_33, 3 - c_44, 4 - density. Array must use 64 bit integers with stiffness tensors in MPa and density in Kg/m^3. To use these values the material index of grid points should be 0. If a point is not using stiffness tensors and density the values are not used. If parameter not used, velocity curves will be used instead(don't set material index to 0).
        :param n_threads_TTF: Number of threads to be used for calculating travel time fields.
        :param n_threads_rays: Number of threads to be used for calculating travel time fields.
        :param low_mem: Boolean value to reduce ram usage. False will return the travel time fields and True will save to the directory as "temp_TTF_i.npy" for index i. Can be loaded using TTF = np.load("temp_TTF_i.npy"). If True then the number of threads must be greater than 1 (Only saves memory for lots of sources).
        :return: 2D numpy array of travel times along the ray paths. When the path is not calculated the value is 0.
        """

        min_vel, max_vel = min_max_vel(veln, velpn, vel_map, stif_den, self.velocity_dat)
        if min_vel < 1000:
            Warning(f"Minimum velocity is {min_vel}. Model may be input incorrectly")
        if max_vel > 15000:
            Warning(f"Maximum velocity is {max_vel}. Model may be input incorrectly")

        # If no velocity scaling is required.
        if type(vel_map) == type(None):
            vel_map = np.ones(veln.shape)
        # If no stifness tensors and density are being used.
        if type(stif_den) == type(None):
            stif_den = np.zeros((veln.shape[0], veln.shape[1], 5), dtype=np.int64)
        # Set up arrays for storing ray paths in the class.
        n_trans = len(self.isx)
        if save_rays:
            self.ray_paths_x = np.zeros((n_trans, n_trans, 5 * (veln.shape[0] + veln.shape[1])))
            self.ray_paths_y = np.copy(self.ray_paths_x)
            self.ray_len = np.zeros((n_trans, n_trans), dtype=int)

        if type(trans_pairs) == type(None):
            # If the ray paths that are required are not included then all combinations of transducers are used (each pair only uses one ray path i.e. transducer in each pair is either source or reciever.
            trans_pairs = np.zeros((n_trans, n_trans))
            for i in range(n_trans):
                for j in range(n_trans):
                    if i < j:
                        trans_pairs[i, j] = 1

        # Determine which travel time fields needs calculating
        rec_trans = np.zeros(n_trans)
        for j in range(n_trans):
            if sum(trans_pairs[:, j]) > 0:
                rec_trans[j] = 1
        #print(trans_pairs)
        #print(f"total: {np.sum(trans_pairs)}")
        #print(rec_trans)

        # Calculate required travel time fields
        if n_threads_TTF == 1:
            TTFs = self.update(veln, velpn, vel_map, stif_den, subgrid_size, rec_trans)
        else:
            if low_mem == True:
                self.update_parallel(veln, velpn, vel_map, stif_den, subgrid_size, rec_trans, n_threads=n_threads_TTF, low_mem=True)
                TTFs = None
            else:
                TTFs = self.update_parallel(veln, velpn, vel_map, stif_den, subgrid_size, rec_trans, n_threads=n_threads_TTF, low_mem=False)
        #for i in range(n_trans):
        #    plt.contourf(TTFs[i, :, :], 20)
        #    plt.show()

        # Find transducer positions on finer grid.
        new_trans_x = subgrid_size * self.isx
        new_trans_y = subgrid_size * self.isz
        #print(self.isx, self.isz)
        #print(new_trans_x, new_trans_y)
        if type(trans_pairs) == type(None):
            trans_pairs = np.ones((n_trans, n_trans))
        # Array for storing ray times
        times = np.zeros((n_trans, n_trans))


        if n_threads_rays == 1:
            for i in range(n_trans):
                for j in range(n_trans): #for j in range(i + 1, len(self.isx)):
                    if i != j:
                        if trans_pairs[i, j] == 1:
                            # If ray path is required.
                            source = np.array([new_trans_x[i], new_trans_y[i]])
                            receiver = np.array([new_trans_x[j], new_trans_y[j]])
                            ray_x, ray_y, times[i, j] = find_ray(self.dnx, self.velocity_dat, source, receiver, TTFs[j, :, :], veln, velpn, vel_map, stif_den, subgrid_size)

                            # Move ray paths from finer grid back to original grid.
                            ray_x = ray_x / subgrid_size
                            ray_y = ray_y / subgrid_size

                            # Store ray path into class to be retrieved using class function ray_path
                            if save_rays:
                                ray_len = len(ray_x)
                                self.ray_paths_x[i, j, 0:ray_len] = ray_x
                                self.ray_paths_y[i, j, 0:ray_len] = ray_y
                                self.ray_len[i, j] = ray_len
        else:
            # Queue for sending data between processes.
            queue1 = multiprocessing.Queue()
            queue2 = multiprocessing.Queue()
            queue3 = multiprocessing.Queue()
            for i in range(n_trans):
                for j in range(n_trans):
                    if i != j:
                        if trans_pairs[i, j] == 1:
                            # add all required jobs into queue to be retrieved by other processes
                            source = np.array([new_trans_x[i], new_trans_y[i]])
                            receiver = np.array([new_trans_x[j], new_trans_y[j]])
                            queue1.put([i, j, source, receiver])
            # Set up worker processes
            processes = []
            for i in range(n_threads_rays):
                process = multiprocessing.Process(target=parallel_rays, args=(i, queue1, queue2, queue3, self.dnx, self.velocity_dat, TTFs, veln, velpn, vel_map, stif_den, subgrid_size))
                processes.append(process)
            # Start processes running
            for process in processes:
                process.start()
            # run until the correct number of ray paths has been retrieved.
            worker_jobs = np.zeros((n_threads_rays, 2), dtype=int)
            for ray_num in tqdm(range(int(np.sum(trans_pairs))), disable=tqdm_disable, ncols=100, desc="Finished ray paths", colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]"):
                #print(f"rays completed {ray_num}/{int(np.sum(trans_pairs))}")
                #print(f"{int(np.sum(trans_pairs))} , {ray_num}, {n_threads_rays}, {(int(np.sum(trans_pairs)) - ray_num) > n_threads_rays}")
                while queue3.empty() == False:
                    [thread_num, ind_1, ind_2] = queue3.get()
                    worker_jobs[thread_num, :] = [ind_1, ind_2]
                if (int(np.sum(trans_pairs)) - ray_num) > n_threads_rays:  # If almost finished main process checks for crashes periodically else checks when data is received
                    # Retrieve ray path and time from worker processes.
                    [i, j, ray_x, ray_y, time] = queue2.get()
                    # Store ray path into class to be retrieved using class function ray_path.
                    if save_rays:
                        ray_len = len(ray_x)
                        self.ray_paths_x[i, j, 0:ray_len] = ray_x
                        self.ray_paths_y[i, j, 0:ray_len] = ray_y
                        self.ray_len[i, j] = ray_len
                    times[i, j] = time

                    #activ_proc = np.zeros(n_threads_rays, dtype=bool)
                    for proc_num in range(n_threads_rays):
                        #activ_proc[proc_num] = processes[proc_num].is_alive()
                        if processes[proc_num].is_alive() == False:
                            print(f"Process {proc_num} failed on ray [i,j] = {worker_jobs[proc_num, :]} with exit code {processes[proc_num].exitcode}\nRestarting Process\n", end="")
                            ind_1, ind_2 = worker_jobs[proc_num, :]
                            source = np.array([new_trans_x[ind_1], new_trans_y[ind_1]])
                            receiver = np.array([new_trans_x[ind_2], new_trans_y[ind_2]])
                            [ind_1, ind_2] = worker_jobs[proc_num, :]
                            queue1.put([int(ind_1), int(ind_2), source, receiver])
                            processes[proc_num] = multiprocessing.Process(target=parallel_rays, args=(proc_num, queue1, queue2, queue3, self.dnx, self.velocity_dat, TTFs, veln, velpn, vel_map, stif_den, subgrid_size))
                            processes[proc_num].start()
                    #print(f"{np.sum(activ_proc)}/{n_threads_rays} Alive Processes :{activ_proc}")
                else:
                    while queue2.empty() == True:
                        for proc_num in range(n_threads_rays):
                            if processes[proc_num].is_alive() == False:
                                print(f"Process {proc_num} failed on ray [i,j] = {worker_jobs[proc_num, :]} with exit code {processes[proc_num].exitcode}\nRestarting Process\n", end="")
                                ind_1, ind_2 = worker_jobs[proc_num, :]
                                source = np.array([new_trans_x[ind_1], new_trans_y[ind_1]])
                                receiver = np.array([new_trans_x[ind_2], new_trans_y[ind_2]])
                                [ind_1, ind_2] = worker_jobs[proc_num, :]
                                queue1.put([int(ind_1), int(ind_2), source, receiver])
                                processes[proc_num] = multiprocessing.Process(target=parallel_rays, args=(proc_num, queue1, queue2, queue3, self.dnx, self.velocity_dat, TTFs, veln, velpn, vel_map, stif_den, subgrid_size))
                                processes[proc_num].start()
                        sleep(0.5)
                    [i, j, ray_x, ray_y, time] = queue2.get()
                    # Store ray path into class to be retrieved using class function ray_path.
                    if save_rays:
                        ray_len = len(ray_x)
                        self.ray_paths_x[i, j, 0:ray_len] = ray_x
                        self.ray_paths_y[i, j, 0:ray_len] = ray_y
                        self.ray_len[i, j] = ray_len
                    times[i, j] = time

            # Terminate worker processes.
            for process in processes:
                process.terminate()
            # Wait for worker processes have finished terminating.
            for process in processes:
                process.join()
            # Close worker processes and return memory back to the os.
            for process in processes:
                process.close()
        return times
    '''

    def find_all_TTF_rays_parallel(self, veln, velpn, vel_map=None, subgrid_size=9, trans_pairs=None, stif_den=None, n_threads=2, save_rays=True):
        """
        Computes travel time fields for required receivers and performs ray tracing between source and receiver pairs using parallelisation. Returns travel times, however ray paths can be obtained using the function ray_path. Each ray path is calculated in the same process as the receiver travel time field.

        :param veln: Anisotropic material orientations at grid points. Set as an array of zeros if using isotropic materials.
        :type veln: 2D numpy array
        :param velpn: Material index of each grid point (0 if using stiffness tensors and density, otherwise index for column in velocity table).
        :type velpn: 2D numpy array of type int
        :param vel_map: Velocity scaling parameters for each grid point. Use array of ones for anisotropic materials, or array of velocities for isotropic materials. If unused then a array of ones is used.
        :type vel_map: 2D numpy array
        :param subgrid_size: Parameter for computing travel time field on finer grid (must be a odd number), multiply indices by subgrid_size to move between indices on original grid. Default value is set to 1 i.e same as original grid.
        :type subgrid_size: int
        :param trans_pairs: Transducer pairs where ray tracing is being performed (0 - No ray, 1 - compute ray). If parameter not used then all rays calculated(only one ray calculated per transducer pair i.e transducer is either source or receiver). Leading diagonal must have zeros.
        :type trans_pairs: 2D numpy array
        :param stif_den: Stiffness tensors at each grid point. First two indices are the position of the grid point and 3rd index is for the materials parameters, 0 - c_22, 1 - c_23, 2 - c_33, 3 - c_44, 4 - density. Array must use 64 bit integers with stiffness tensors in MPa and density in Kg/m^3. To use these values the material index of grid points should be 0. If a point is not using stiffness tensors and density the values are not used. If parameter not used, velocity curves will be used instead(don't set material index to 0).
        :type stif_den: 3D numpy array of type np.int64
        :param n_threads: Number of threads to be used.
        :type n_threads: int
        :param save_rays: Boolean value for if rays should be saved into class (rays can be obtained using ray_path function).
        :type save_rays: bool
        :return: Travel times along the ray paths. When the path is not calculated the value is 0.
        :rtype: 2D numpy array
        """
        if n_threads == 1:
            raise ValueError("n_threads should not equal one. Use find_all_TTF_rays for single process.")

        # If no velocity scaling is required.
        if type(vel_map) == type(None):
            vel_map = np.ones(veln.shape)
        # If no stifness tensors and density are being used.
        if type(stif_den) == type(None):
            stif_den = np.zeros((veln.shape[0], veln.shape[1], 5), dtype=np.int64)

        min_vel, max_vel = min_max_vel(veln, velpn, vel_map, stif_den, self.velocity_dat)
        if min_vel < 1000:
            Warning(f"Minimum velocity is {min_vel}. Model may be input incorrectly")
        if max_vel > 15000:
            Warning(f"Maximum velocity is {max_vel}. Model may be input incorrectly")

        # Set up arrays for storing ray paths in the class.
        n_trans = len(self.isx)
        if save_rays:
            self.ray_paths_x = np.zeros((n_trans, n_trans, 5 * (veln.shape[0] + veln.shape[1])))
            self.ray_paths_y = np.copy(self.ray_paths_x)
            self.ray_len = np.zeros((n_trans, n_trans), dtype=int)

        if type(trans_pairs) == type(None):
            # If the ray paths that are required are not included then all combinations of transducers are used (each pair only uses one ray path i.e. transducer in each pair is either source or reciever.
            trans_pairs = np.zeros((n_trans, n_trans))
            for i in range(n_trans):
                for j in range(n_trans):
                    if i < j:
                        trans_pairs[i, j] = 1

        # Determine which travel time fields needs calculating
        rec_trans = np.zeros(n_trans)
        for j in range(n_trans):
            if sum(trans_pairs[:, j]) > 0:
                rec_trans[j] = 1
        #print(trans_pairs)
        #print(f"total: {np.sum(trans_pairs)}")
        #print(rec_trans)

        # Calculate required travel time fields

        #for i in range(n_trans):
        #    plt.contourf(TTFs[i, :, :], 20)
        #    plt.show()

        # Find transducer positions on finer grid.
        new_trans_x = subgrid_size * self.isx
        new_trans_y = subgrid_size * self.isz
        #print(self.isx, self.isz)
        #print(new_trans_x, new_trans_y)
        if type(trans_pairs) == type(None):
            trans_pairs = np.zeros((n_trans, n_trans))
            for i in range(n_trans):
                for j in range(n_trans):
                    if i < j:
                        trans_pairs[i, j] = 1
        n_rays = np.sum(trans_pairs)
        # Array for storing ray times
        times = np.zeros((n_trans, n_trans))

        rays_comp = np.zeros((n_trans, n_trans), dtype=int)


        # Queue for sending data between processes.
        queue1 = multiprocessing.Queue()
        queue2 = multiprocessing.Queue()
        queue3 = multiprocessing.Queue()
        for i in range(n_trans):
            if rec_trans[i] == 1:
                queue1.put(i)

        pbar_TTF = tqdm(total=int(np.sum(rec_trans)), disable=tqdm_disable, ncols=100, desc="Finished TTF's    ", colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]")
        pbar_rays = tqdm(total=n_rays, disable=tqdm_disable, ncols=100, desc="Finished ray paths", colour="green", bar_format="{l_bar} {bar} | {n_fmt}/{total_fmt} [{elapsed}]")


        # Set up worker processes
        processes = []
        for i in range(n_threads):
            process = multiprocessing.Process(target=parallel_TTF_rays, args=(i, queue1, queue2, trans_pairs, veln, velpn, vel_map, stif_den, subgrid_size, self.velocity_dat, self.phase_vel, self.gox, self.goz, self.dnx, self.scx, self.scz, new_trans_x, new_trans_y))
            # Start processes running
            process.start()
            processes.append(process)

        # run until the correct number of ray paths has been retrieved.
        n_rays_comp = 0
        while n_rays_comp < n_rays:
            [return_code, data] = queue2.get()
            if return_code == 0:
                pbar_TTF.update(1)
            else:
                [i, j, ray_x, ray_y, time] = data
                #np.save(f"ray_x_{i}_{j}.npy", ray_x)
                #np.save(f"ray_y_{i}_{j}.npy", ray_y)
                if save_rays:
                    ray_len = len(ray_x)
                    self.ray_paths_x[i, j, 0:ray_len] = ray_x
                    self.ray_paths_y[i, j, 0:ray_len] = ray_y
                    self.ray_len[i, j] = ray_len
                times[i, j] = time
                pbar_rays.update(1)
                n_rays_comp += 1

        # Terminate worker processes.
        for process in processes:
            process.terminate()
        # Wait for worker processes have finished terminating.
        for process in processes:
            process.join()
        # Close worker processes and return memory back to the os.
        for process in processes:
            process.close()
        return times

    def ray_path(self, i, j):
        """
        Function for returning the ray path for a given source and receiver which was calculated in the function find_all_TTF_rays. Will return None values if the path hasn't been calculated.

        :param i: Source index for the required ray path.
        :type i: int
        :param j: Reciever index for the required ray path.
        :type j: int
        :return: ray_x, ray_y - Arrays for the x and y positions of points in the ray path. If there is no path then None values are returned.
        :rtype: 1D numpy array, 1D numpy array
        """
        # Check if ray path exists.
        if self.ray_len[i, j] == 0:
            print("Ray path has not been calculated")
            return None, None
        else:
            # Return ray path
            ray_len = self.ray_len[i, j]
            return self.ray_paths_x[i, j, 0:ray_len], self.ray_paths_y[i, j, 0:ray_len]









