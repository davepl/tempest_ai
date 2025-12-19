#!/usr/bin/env python3
"""
Render a short animation of a wireframe torus made of box-like segments.
The rotation plus lack of shading yields the “could be rotating either way” effect.
"""
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def rotation_matrix_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def make_box_vertices(
    angle_center: float,
    angle_span: float,
    r_inner: float,
    r_outer: float,
    height: float,
) -> np.ndarray:
    """Return 8 vertices for a box segment on a torus ring."""
    half_span = angle_span * 0.5
    angles = [angle_center - half_span, angle_center + half_span]
    radii = [r_inner, r_outer]
    zs = [-height * 0.5, height * 0.5]

    verts = []
    for ang in angles:
        ca, sa = math.cos(ang), math.sin(ang)
        for r in radii:
            for z in zs:
                verts.append([r * ca, r * sa, z])
    return np.array(verts, dtype=float)


def box_faces_from_vertices(v: np.ndarray):
    """Indices defining the 6 faces of the box (quads)."""
    # Vertex order generated in make_box_vertices:
    # angle0-r_inner (z-), angle0-r_inner (z+), angle0-r_outer (z-), angle0-r_outer (z+),
    # angle1-r_inner (z-), angle1-r_inner (z+), angle1-r_outer (z-), angle1-r_outer (z+)
    return [
        [0, 1, 3, 2],  # inner-angle side
        [4, 5, 7, 6],  # outer-angle side
        [0, 2, 6, 4],  # radial inner face
        [1, 3, 7, 5],  # radial outer face
        [0, 1, 5, 4],  # bottom (z-)
        [2, 3, 7, 6],  # top (z+)
    ]


def generate_stars(num: int, r_inner: float, r_outer: float, height: float) -> np.ndarray:
    """Scatter small white points within the torus volume."""
    angles = np.random.uniform(0, 2 * math.pi, size=num)
    radii = np.sqrt(np.random.uniform(r_inner * r_inner, r_outer * r_outer, size=num))
    zs = np.random.uniform(-height * 0.5, height * 0.5, size=num)
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    return np.stack([xs, ys, zs], axis=1)


def main():
    out_dir = Path("renders")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_path = out_dir / "rotating_torus_boxes.mp4"

    n_segments = 16
    r_inner = 4.5
    r_outer = 6.2
    seg_height = 2.2
    angle_span = (2 * math.pi / n_segments) * 0.85

    # Precompute segment vertices
    segment_vertices = []
    for i in range(n_segments):
        ang = (2 * math.pi / n_segments) * i
        v = make_box_vertices(ang, angle_span, r_inner, r_outer, seg_height)
        segment_vertices.append(v)

    stars = generate_stars(2200, r_inner * 0.75, r_outer * 1.05, seg_height * 1.2)

    fig = plt.figure(figsize=(6, 6), facecolor="black", dpi=150)
    ax = fig.add_subplot(111, projection="3d", facecolor="black")
    ax.set_axis_off()
    lim = r_outer * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim * 0.6, lim * 0.6)
    ax.view_init(elev=25, azim=30)

    # Collections for update
    face_collections = []
    edge_collections = []
    for _ in range(n_segments):
        poly = Poly3DCollection([], facecolor=(0, 0, 0, 0), edgecolor="white", linewidth=0.6)
        ax.add_collection3d(poly)
        face_collections.append(poly)

        edges = Line3DCollection([], colors="white", linewidths=0.8)
        ax.add_collection3d(edges)
        edge_collections.append(edges)

    stars_scatter = ax.scatter(
        stars[:, 0],
        stars[:, 1],
        stars[:, 2],
        s=0.15,
        c="white",
        alpha=0.8,
        depthshade=False,
    )

    def update(frame_idx: int):
        theta = frame_idx * (2 * math.pi / 240.0)  # full rotation in 240 frames
        rot = rotation_matrix_z(theta)
        rotated_stars = stars @ rot.T
        stars_scatter._offsets3d = (
            rotated_stars[:, 0],
            rotated_stars[:, 1],
            rotated_stars[:, 2],
        )

        for i, base_v in enumerate(segment_vertices):
            v = base_v @ rot.T
            faces_idx = box_faces_from_vertices(v)
            faces = [[v[idx] for idx in face] for face in faces_idx]
            face_collections[i].set_verts(faces)

            # Edge list
            edges = []
            for f in faces_idx:
                for a, b in zip(f, f[1:] + [f[0]]):
                    edges.append([v[a], v[b]])
            edge_collections[i].set_segments(edges)

        # Slight camera orbit to enhance direction ambiguity
        ax.view_init(elev=25, azim=30 + frame_idx * 0.5)
        return face_collections + edge_collections + [stars_scatter]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=240,
        interval=33,
        blit=False,
    )

    print(f"Writing animation to {out_path} ...")
    ani.save(out_path, fps=30, dpi=150, codec="h264", bitrate=8000)
    print("Done.")


if __name__ == "__main__":
    main()
