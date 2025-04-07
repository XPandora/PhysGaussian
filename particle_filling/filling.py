import torch
import os
import numpy as np
import taichi as ti
import mcubes

# 1. densify grids
# 2. identify grids whose density is larger than some threshold
# 3. filling grids with particles
# 4. identify and fill internal grids


@ti.func
def compute_density(index, pos, opacity, cov, grid_dx):
    gaussian_weight = 0.0
    for i in range(0, 2):
        for j in range(0, 2):
            for k in range(0, 2):
                node_pos = (index + ti.Vector([i, j, k])) * grid_dx
                dist = pos - node_pos
                gaussian_weight += ti.exp(-0.5 * dist.dot(cov @ dist))

    return opacity * gaussian_weight / 8.0


@ti.kernel
def densify_grids(
    init_particles: ti.template(),
    opacity: ti.template(),
    cov_upper: ti.template(),
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
):
    for pi in range(init_particles.shape[0]):
        pos = init_particles[pi]
        x = pos[0]
        y = pos[1]
        z = pos[2]
        i = ti.floor(x / grid_dx, dtype=int)
        j = ti.floor(y / grid_dx, dtype=int)
        k = ti.floor(z / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)
        cov = ti.Matrix(
            [
                [cov_upper[pi][0], cov_upper[pi][1], cov_upper[pi][2]],
                [cov_upper[pi][1], cov_upper[pi][3], cov_upper[pi][4]],
                [cov_upper[pi][2], cov_upper[pi][4], cov_upper[pi][5]],
            ]
        )
        sig, Q = ti.sym_eig(cov)
        sig[0] = ti.max(sig[0], 1e-8)
        sig[1] = ti.max(sig[1], 1e-8)
        sig[2] = ti.max(sig[2], 1e-8)
        sig_mat = ti.Matrix(
            [[1.0 / sig[0], 0, 0], [0, 1.0 / sig[1], 0], [0, 0, 1.0 / sig[2]]]
        )
        cov = Q @ sig_mat @ Q.transpose()
        r = 0.0
        for idx in ti.static(range(3)):
            if sig[idx] < 0:
                sig[idx] = ti.sqrt(-sig[idx])
            else:
                sig[idx] = ti.sqrt(sig[idx])

            r = ti.max(r, sig[idx])

        r = ti.ceil(r / grid_dx, dtype=int)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if (
                        i + dx >= 0
                        and i + dx < grid_density.shape[0]
                        and j + dy >= 0
                        and j + dy < grid_density.shape[1]
                        and k + dz >= 0
                        and k + dz < grid_density.shape[2]
                    ):
                        density = compute_density(
                            ti.Vector([i + dx, j + dy, k + dz]),
                            pos,
                            opacity[pi],
                            cov,
                            grid_dx,
                        )
                        ti.atomic_add(grid_density[i + dx, j + dy, k + dz], density)


@ti.kernel
def fill_dense_grids(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    density_thres: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
) -> int:
    new_start_idx = start_idx
    for i, j, k in grid_density:
        if grid_density[i, j, k] > density_thres:
            if grid[i, j, k] < max_particles_per_cell:
                diff = max_particles_per_cell - grid[i, j, k]
                grid[i, j, k] = max_particles_per_cell
                tmp_start_idx = ti.atomic_add(new_start_idx, diff)

                for index in range(tmp_start_idx, tmp_start_idx + diff):
                    di = ti.random()
                    dj = ti.random()
                    dk = ti.random()
                    new_particles[index] = ti.Vector([i + di, j + dj, k + dk]) * grid_dx

    return new_start_idx


@ti.func
def collision_search(
    grid: ti.template(), grid_density: ti.template(), index, dir_type, size, threshold
) -> bool:
    dir = ti.Vector([0, 0, 0])
    if dir_type == 0:
        dir[0] = 1
    elif dir_type == 1:
        dir[0] = -1
    elif dir_type == 2:
        dir[1] = 1
    elif dir_type == 3:
        dir[1] = -1
    elif dir_type == 4:
        dir[2] = 1
    elif dir_type == 5:
        dir[2] = -1

    flag = False
    index += dir
    i, j, k = index
    while ti.max(i, j, k) < size and ti.min(i, j, k) >= 0:
        if grid_density[index] > threshold:
            flag = True
            break
        index += dir
        i, j, k = index

    return flag


@ti.func
def collision_times(
    grid: ti.template(), grid_density: ti.template(), index, dir_type, size, threshold
) -> int:
    dir = ti.Vector([0, 0, 0])
    times = 0
    if dir_type > 5 or dir_type < 0:
        times = 1
    else:
        if dir_type == 0:
            dir[0] = 1
        elif dir_type == 1:
            dir[0] = -1
        elif dir_type == 2:
            dir[1] = 1
        elif dir_type == 3:
            dir[1] = -1
        elif dir_type == 4:
            dir[2] = 1
        elif dir_type == 5:
            dir[2] = -1

        state = grid[index] > 0
        index += dir
        i, j, k = index
        while ti.max(i, j, k) < size and ti.min(i, j, k) >= 0:
            new_state = grid_density[index] > threshold
            if new_state != state and state == False:
                times += 1
            state = new_state
            index += dir
            i, j, k = index

    return times


@ti.kernel
def internal_filling(
    grid: ti.template(),
    grid_density: ti.template(),
    grid_dx: float,
    new_particles: ti.template(),
    start_idx: int,
    max_particles_per_cell: int,
    exclude_dir: int,
    ray_cast_dir: int,
    threshold: float,
) -> int:
    new_start_idx = start_idx
    for i, j, k in grid:
        if grid[i, j, k] == 0:
            collision_hit = True
            for dir_type in ti.static(range(6)):
                if dir_type != exclude_dir:
                    hit_test = collision_search(
                        grid=grid,
                        grid_density=grid_density,
                        index=ti.Vector([i, j, k]),
                        dir_type=dir_type,
                        size=grid.shape[0],
                        threshold=threshold,
                    )
                    collision_hit = collision_hit and hit_test

            if collision_hit:
                hit_times = collision_times(
                    grid=grid,
                    grid_density=grid_density,
                    index=ti.Vector([i, j, k]),
                    dir_type=ray_cast_dir,
                    size=grid.shape[0],
                    threshold=threshold,
                )

                if ti.math.mod(hit_times, 2) == 1:
                    diff = max_particles_per_cell - grid[i, j, k]
                    grid[i, j, k] = max_particles_per_cell
                    tmp_start_idx = ti.atomic_add(new_start_idx, diff)
                    for index in range(tmp_start_idx, tmp_start_idx + diff):
                        di = ti.random()
                        dj = ti.random()
                        dk = ti.random()
                        new_particles[index] = (
                            ti.Vector([i + di, j + dj, k + dk]) * grid_dx
                        )

    return new_start_idx


@ti.kernel
def assign_particle_to_grid(pos: ti.template(), grid: ti.template(), grid_dx: float):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        ti.atomic_add(grid[i, j, k], 1)


@ti.kernel
def compute_particle_volume(
    pos: ti.template(), grid: ti.template(), particle_vol: ti.template(), grid_dx: float
):
    for pi in range(pos.shape[0]):
        p = pos[pi]
        i = ti.floor(p[0] / grid_dx, dtype=int)
        j = ti.floor(p[1] / grid_dx, dtype=int)
        k = ti.floor(p[2] / grid_dx, dtype=int)
        particle_vol[pi] = (grid_dx * grid_dx * grid_dx) / grid[i, j, k]


def get_particle_volume(pos, grid_n: int, grid_dx: float, unifrom: bool = False):
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    particle_vol = ti.field(dtype=float, shape=pos.shape[0])

    assign_particle_to_grid(ti_pos, grid, grid_dx)
    compute_particle_volume(ti_pos, grid, particle_vol, grid_dx)

    if unifrom:
        vol = particle_vol.to_torch()
        vol = torch.mean(vol).repeat(pos.shape[0])
        return vol
    else:
        return particle_vol.to_torch()


def fill_particles(
    pos,
    opacity,
    cov,
    grid_n: int,
    max_samples: int,
    grid_dx: float,
    density_thres=2.0,
    search_thres=1.0,
    max_particles_per_cell=1,
    search_exclude_dir=5,
    ray_cast_dir=4,
    boundary: list = None,
    smooth: bool = False,
):
    pos_clone = pos.clone()
    if boundary is not None:
        assert len(boundary) == 6
        mask = torch.ones(pos_clone.shape[0], dtype=torch.bool).cuda()
        max_diff = 0.0
        for i in range(3):
            mask = torch.logical_and(mask, pos_clone[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, pos_clone[:, i] < boundary[2 * i + 1])
            max_diff = max(max_diff, boundary[2 * i + 1] - boundary[2 * i])

        pos = pos[mask]
        opacity = opacity[mask]
        cov = cov[mask]

        grid_dx = max_diff / grid_n
        new_origin = torch.tensor([boundary[0], boundary[2], boundary[4]]).cuda()
        pos = pos - new_origin

    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_opacity.from_torch(opacity.reshape(-1))
    ti_cov.from_torch(cov.reshape(-1, 6))

    grid = ti.field(dtype=int, shape=(grid_n, grid_n, grid_n))
    grid_density = ti.field(dtype=float, shape=(grid_n, grid_n, grid_n))
    particles = ti.Vector.field(n=3, dtype=float, shape=max_samples)
    fill_num = 0

    # compute density_field
    densify_grids(ti_pos, ti_opacity, ti_cov, grid, grid_density, grid_dx)

    # fill dense grids
    fill_num = fill_dense_grids(
        grid,
        grid_density,
        grid_dx,
        density_thres,
        particles,
        0,
        max_particles_per_cell,
    )
    print("after dense grids: ", fill_num)

    # smooth density_field
    if smooth:
        df = grid_density.to_numpy()
        smoothed_df = mcubes.smooth(df, method="constrained", max_iters=500).astype(
            np.float32
        )
        grid_density.from_numpy(smoothed_df)
        print("smooth finished")

    # fill internal grids
    fill_num = internal_filling(
        grid,
        grid_density,
        grid_dx,
        particles,
        fill_num,
        max_particles_per_cell,
        exclude_dir=search_exclude_dir,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        ray_cast_dir=ray_cast_dir,  # 0: x, 1: -x, 2: y, 3: -y, 4: z, 5: -z direction
        threshold=search_thres,
    )
    print("after internal grids: ", fill_num)

    # put new particles together with original particles
    particles_tensor = particles.to_torch()[:fill_num].cuda()
    if boundary is not None:
        particles_tensor = particles_tensor + new_origin
    particles_tensor = torch.cat([pos_clone, particles_tensor], dim=0)

    return particles_tensor


@ti.kernel
def get_attr_from_closest(
    ti_pos: ti.template(),
    ti_shs: ti.template(),
    ti_opacity: ti.template(),
    ti_cov: ti.template(),
    ti_new_pos: ti.template(),
    ti_new_shs: ti.template(),
    ti_new_opacity: ti.template(),
    ti_new_cov: ti.template(),
):
    for pi in range(ti_new_pos.shape[0]):
        p = ti_new_pos[pi]
        min_dist = 1e10
        min_idx = -1
        for pj in range(ti_pos.shape[0]):
            dist = (p - ti_pos[pj]).norm()
            if dist < min_dist:
                min_dist = dist
                min_idx = pj
        ti_new_shs[pi] = ti_shs[min_idx]
        ti_new_opacity[pi] = ti_opacity[min_idx]
        ti_new_cov[pi] = ti_cov[min_idx]


def init_filled_particles(pos, shs, cov, opacity, new_pos):
    shs = shs.reshape(pos.shape[0], -1)
    ti_pos = ti.Vector.field(n=3, dtype=float, shape=pos.shape[0])
    ti_cov = ti.Vector.field(n=6, dtype=float, shape=cov.shape[0])
    ti_shs = ti.Vector.field(n=shs.shape[1], dtype=float, shape=shs.shape[0])
    ti_opacity = ti.field(dtype=float, shape=opacity.shape[0])
    ti_pos.from_torch(pos.reshape(-1, 3))
    ti_cov.from_torch(cov.reshape(-1, 6))
    ti_shs.from_torch(shs)
    ti_opacity.from_torch(opacity.reshape(-1))

    new_shs = torch.mean(shs, dim=0).repeat(new_pos.shape[0], 1).cuda()
    ti_new_pos = ti.Vector.field(n=3, dtype=float, shape=new_pos.shape[0])
    ti_new_shs = ti.Vector.field(n=shs.shape[1], dtype=float, shape=new_pos.shape[0])
    ti_new_opacity = ti.field(dtype=float, shape=new_pos.shape[0])
    ti_new_cov = ti.Vector.field(n=6, dtype=float, shape=new_pos.shape[0])
    ti_new_pos.from_torch(new_pos.reshape(-1, 3))
    ti_new_shs.from_torch(new_shs)

    get_attr_from_closest(
        ti_pos,
        ti_shs,
        ti_opacity,
        ti_cov,
        ti_new_pos,
        ti_new_shs,
        ti_new_opacity,
        ti_new_cov,
    )

    shs_tensor = ti_new_shs.to_torch().cuda()
    opacity_tensor = ti_new_opacity.to_torch().cuda()
    cov_tensor = ti_new_cov.to_torch().cuda()

    shs_tensor = torch.cat([shs, shs_tensor], dim=0)
    shs_tensor = shs_tensor.view(shs_tensor.shape[0], -1, 3)
    opacity_tensor = torch.cat([opacity, opacity_tensor.reshape(-1, 1)], dim=0)
    cov_tensor = torch.cat([cov, cov_tensor], dim=0)
    return shs_tensor, opacity_tensor, cov_tensor
