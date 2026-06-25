"""2D centerline extraction for a soft manipulator image.

This implements the vision-only part of the ASES pipeline described in
Zou et al. (2022): color segmentation, morphology, contour extraction, and
1D SOM centerline estimation. A single RGB image cannot provide stereo 3D
reconstruction, so the output is an ordered 2D backbone point set.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_IMAGE = Path(__file__).with_name("soro.png")


@dataclass(frozen=True)
class HSVRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass
class CenterlineResult:
    image: np.ndarray
    mask: np.ndarray
    edges: np.ndarray
    contour_points: np.ndarray
    som_points: np.ndarray
    skeleton_points: np.ndarray


HSV_PRESETS: dict[str, HSVRange] = {
    # The provided soro.png has a dark blue/purple soft manipulator body.
    "blue": HSVRange((90, 35, 20), (145, 255, 210)),
    # Useful when the target body/marker is orange in the paper-like setup.
    "orange": HSVRange((0, 60, 60), (30, 255, 255)),
    "green": HSVRange((35, 40, 40), (85, 255, 255)),
    "red1": HSVRange((0, 60, 40), (12, 255, 255)),
    "red2": HSVRange((165, 60, 40), (179, 255, 255)),
}


def parse_hsv_triplet(value: str) -> tuple[int, int, int]:
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("HSV values must be formatted as H,S,V")
    h, s, v = parts
    if not (0 <= h <= 179 and 0 <= s <= 255 and 0 <= v <= 255):
        raise argparse.ArgumentTypeError("HSV ranges are H:0-179, S:0-255, V:0-255")
    return h, s, v


def build_mask(
    image_bgr: np.ndarray,
    hsv_range: HSVRange,
    gaussian_kernel: int = 7,
    close_kernel: int = 19,
    open_kernel: int = 5,
) -> np.ndarray:
    """Segment the manipulator by HSV threshold and morphology."""

    gaussian_kernel = max(3, gaussian_kernel | 1)
    close_kernel = max(3, close_kernel | 1)
    open_kernel = max(3, open_kernel | 1)

    blurred = cv2.GaussianBlur(image_bgr, (gaussian_kernel, gaussian_kernel), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_range.lower), np.array(hsv_range.upper))

    open_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
    )
    close_element = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_element, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_element, iterations=2)
    mask = keep_largest_component(mask)
    mask = fill_holes(mask)
    return mask


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest foreground connected component."""

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background holes inside the segmented component."""

    flood = mask.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def extract_contour_points(
    mask: np.ndarray, canny_low: int, canny_high: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return edge image and contour point cloud as xy coordinates."""

    edges = cv2.Canny(mask, canny_low, canny_high)
    yx = np.column_stack(np.where(edges > 0))
    if yx.size == 0:
        raise RuntimeError("No contour points were detected. Adjust the HSV threshold.")
    xy = yx[:, ::-1].astype(np.float32)
    return edges, xy


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    """Compute a simple binary skeleton using iterative morphology."""

    src = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(src)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(src) > 0:
        eroded = cv2.erode(src, element)
        opened = cv2.dilate(eroded, element)
        residue = cv2.subtract(src, opened)
        skeleton = cv2.bitwise_or(skeleton, residue)
        src = eroded

    return keep_largest_component(skeleton)


def skeleton_main_path(skeleton: np.ndarray) -> np.ndarray:
    """Extract the longest graph path from a skeleton image as ordered xy points."""

    yx = np.column_stack(np.where(skeleton > 0))
    if len(yx) < 2:
        return np.empty((0, 2), dtype=np.float32)

    nodes = [tuple(map(int, p)) for p in yx]
    node_set = set(nodes)
    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in nodes:
        local = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                candidate = (y + dy, x + dx)
                if candidate in node_set:
                    local.append(candidate)
        neighbors[(y, x)] = local

    endpoints = [node for node, local in neighbors.items() if len(local) <= 1]
    if not endpoints:
        endpoints = nodes

    best_start = endpoints[0]
    best_end = endpoints[0]
    best_parent: dict[tuple[int, int], tuple[int, int] | None] = {}
    best_distance = -1

    search_starts = (
        endpoints
        if len(endpoints) <= 80
        else endpoints[:: max(1, len(endpoints) // 80)]
    )
    for start in search_starts:
        distance, parent = bfs_skeleton(start, neighbors)
        reachable_endpoints = [p for p in endpoints if p in distance]
        if not reachable_endpoints:
            reachable_endpoints = list(distance)
        end = max(reachable_endpoints, key=lambda p: distance[p])
        if distance[end] > best_distance:
            best_start = start
            best_end = end
            best_parent = parent
            best_distance = distance[end]

    path = []
    current: tuple[int, int] | None = best_end
    while current is not None:
        path.append(current)
        if current == best_start:
            break
        current = best_parent.get(current)

    path_yx = np.array(path[::-1], dtype=np.float32)
    return path_yx[:, ::-1]


def bfs_skeleton(
    start: tuple[int, int],
    neighbors: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
    distance = {start: 0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        node = queue.popleft()
        for nxt in neighbors[node]:
            if nxt in distance:
                continue
            distance[nxt] = distance[node] + 1
            parent[nxt] = node
            queue.append(nxt)
    return distance, parent


def resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    """Sample count evenly spaced points from an ordered xy polyline."""

    if len(points) == 0:
        raise RuntimeError("Cannot resample an empty polyline.")
    if len(points) == 1:
        return np.repeat(points.astype(np.float32), count, axis=0)

    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(distances)])
    total = cumulative[-1]
    if total == 0:
        return np.repeat(points[:1].astype(np.float32), count, axis=0)

    targets = np.linspace(0.0, total, count)
    sampled = np.empty((count, 2), dtype=np.float32)
    sampled[:, 0] = np.interp(targets, cumulative, points[:, 0])
    sampled[:, 1] = np.interp(targets, cumulative, points[:, 1])
    return sampled


def pca_initial_nodes(points: np.ndarray, count: int) -> np.ndarray:
    """Fallback SOM initialization along the principal axis of the contour cloud."""

    mean = points.mean(axis=0)
    centered = points - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    projected = centered @ axis
    lo, hi = np.percentile(projected, [2, 98])
    values = np.linspace(lo, hi, count)
    return (mean + values[:, None] * axis).astype(np.float32)


def train_som(
    contour_points: np.ndarray,
    initial_nodes: np.ndarray,
    epochs: int = 15,
    alpha0: float = 0.01,
    radius0: float = 3.0,
    seed: int = 7,
    max_points_per_epoch: int = 6000,
) -> np.ndarray:
    """Train an ordered 1D SOM chain on contour points."""

    rng = np.random.default_rng(seed)
    nodes = initial_nodes.astype(np.float32).copy()
    points = contour_points.astype(np.float32)
    node_index = np.arange(len(nodes), dtype=np.float32)

    if len(points) > max_points_per_epoch:
        chosen = rng.choice(len(points), size=max_points_per_epoch, replace=False)
        points = points[chosen]

    total_steps = max(1, epochs * len(points))
    step = 0
    for _ in range(max(1, epochs)):
        order = rng.permutation(len(points))
        for point in points[order]:
            progress = step / total_steps
            alpha = alpha0 * np.exp(-2.0 * progress)
            radius = max(0.5, radius0 * np.exp(-2.0 * progress))

            distances = np.linalg.norm(nodes - point, axis=1)
            winner = int(np.argmin(distances))
            influence = np.exp(-((node_index - winner) ** 2) / (2.0 * radius**2))
            nodes += (alpha * influence[:, None] * (point - nodes)).astype(np.float32)
            step += 1

    return nodes


def estimate_centerline(
    image_path: Path,
    hsv_range: HSVRange,
    backbone_points: int = 7,
    epochs: int = 15,
    alpha0: float = 0.01,
    radius0: float = 3.0,
    canny_low: int = 50,
    canny_high: int = 150,
) -> CenterlineResult:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    mask = build_mask(image, hsv_range)
    edges, contour_points = extract_contour_points(mask, canny_low, canny_high)
    skeleton = morphological_skeleton(mask)
    skeleton_path = skeleton_main_path(skeleton)

    if len(skeleton_path) >= 2:
        initial_nodes = resample_polyline(skeleton_path, backbone_points)
    else:
        initial_nodes = pca_initial_nodes(contour_points, backbone_points)

    som_points = train_som(
        contour_points=contour_points,
        initial_nodes=initial_nodes,
        epochs=epochs,
        alpha0=alpha0,
        radius0=radius0,
    )

    return CenterlineResult(
        image=image,
        mask=mask,
        edges=edges,
        contour_points=contour_points,
        som_points=som_points,
        skeleton_points=skeleton_path,
    )


def save_points_csv(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "x_px", "y_px"])
        for index, (x, y) in enumerate(points):
            writer.writerow([index, f"{x:.3f}", f"{y:.3f}"])


def make_overlay(result: CenterlineResult) -> np.ndarray:
    overlay = result.image.copy()

    edges_yx = np.column_stack(np.where(result.edges > 0))
    if len(edges_yx) > 0:
        overlay[edges_yx[:, 0], edges_yx[:, 1]] = (255, 255, 0)

    points_i = np.round(result.som_points).astype(int)
    points_i[:, 0] = np.clip(points_i[:, 0], 0, overlay.shape[1] - 1)
    points_i[:, 1] = np.clip(points_i[:, 1], 0, overlay.shape[0] - 1)

    cv2.polylines(overlay, [points_i.reshape(-1, 1, 2)], False, (0, 0, 255), 4)
    for index, (x, y) in enumerate(points_i):
        cv2.circle(overlay, (int(x), int(y)), 8, (0, 255, 255), -1)
        cv2.putText(
            overlay,
            str(index),
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return overlay


def make_panel(image: np.ndarray, title: str, panel_size: tuple[int, int]) -> np.ndarray:
    """Create a labeled, letterboxed panel for visual comparison."""

    panel_w, panel_h = panel_size
    title_h = 38
    body_h = panel_h - title_h

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    scale = min(panel_w / w, body_h / h)
    resized_w = max(1, int(round(w * scale)))
    resized_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    panel = np.full((panel_h, panel_w, 3), 245, dtype=np.uint8)
    panel[:title_h, :] = (35, 35, 35)
    x0 = (panel_w - resized_w) // 2
    y0 = title_h + (body_h - resized_h) // 2
    panel[y0 : y0 + resized_h, x0 : x0 + resized_w] = resized

    cv2.putText(
        panel,
        title,
        (16, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (210, 210, 210), 1)
    return panel


def make_debug_grid(result: CenterlineResult) -> np.ndarray:
    """Combine original, mask, edges, and centerline overlay into one image."""

    panel_size = (640, 520)
    panels = [
        make_panel(result.image, "Original", panel_size),
        make_panel(result.mask, "HSV mask", panel_size),
        make_panel(result.edges, "Canny edges", panel_size),
        make_panel(make_overlay(result), "SOM centerline", panel_size),
    ]
    top = np.hstack((panels[0], panels[1]))
    bottom = np.hstack((panels[2], panels[3]))
    return np.vstack((top, bottom))


def save_debug_images(
    output_dir: Path, result: CenterlineResult
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = output_dir / "mask.png"
    edge_path = output_dir / "edges.png"
    overlay_path = output_dir / "centerline_overlay.png"
    grid_path = output_dir / "debug_grid.png"
    cv2.imwrite(str(mask_path), result.mask)
    cv2.imwrite(str(edge_path), result.edges)
    cv2.imwrite(str(overlay_path), make_overlay(result))
    cv2.imwrite(str(grid_path), make_debug_grid(result))
    return mask_path, edge_path, overlay_path, grid_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a 2D soft manipulator centerline with HSV segmentation and SOM."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"Input image path. Defaults to {DEFAULT_IMAGE}",
    )
    parser.add_argument(
        "--target",
        choices=sorted(HSV_PRESETS),
        default="blue",
        help="Preset HSV target color. Use blue for the provided soro.png.",
    )
    parser.add_argument(
        "--hsv-lower", type=parse_hsv_triplet, help="Manual lower HSV as H,S,V."
    )
    parser.add_argument(
        "--hsv-upper", type=parse_hsv_triplet, help="Manual upper HSV as H,S,V."
    )
    parser.add_argument(
        "--points", type=int, default=7, help="Number of backbone/SOM nodes."
    )
    parser.add_argument("--epochs", type=int, default=15, help="SOM training epochs.")
    parser.add_argument(
        "--alpha", type=float, default=0.01, help="Initial SOM learning rate."
    )
    parser.add_argument(
        "--radius", type=float, default=3.0, help="Initial SOM neighborhood radius."
    )
    parser.add_argument(
        "--canny-low", type=int, default=50, help="Canny low threshold."
    )
    parser.add_argument(
        "--canny-high", type=int, default=150, help="Canny high threshold."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("output"),
        help="Directory for CSV and debug images.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Open an OpenCV preview window."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.points < 2:
        raise ValueError("--points must be at least 2.")
    if bool(args.hsv_lower) != bool(args.hsv_upper):
        raise ValueError("--hsv-lower and --hsv-upper must be provided together.")

    hsv_range = (
        HSVRange(args.hsv_lower, args.hsv_upper)
        if args.hsv_lower and args.hsv_upper
        else HSV_PRESETS[args.target]
    )

    result = estimate_centerline(
        image_path=args.image,
        hsv_range=hsv_range,
        backbone_points=args.points,
        epochs=args.epochs,
        alpha0=args.alpha,
        radius0=args.radius,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
    )

    csv_path = args.output_dir / "centerline_points.csv"
    save_points_csv(csv_path, result.som_points)
    mask_path, edge_path, overlay_path, grid_path = save_debug_images(
        args.output_dir, result
    )

    print(f"image: {args.image}")
    print(f"hsv_lower: {hsv_range.lower}")
    print(f"hsv_upper: {hsv_range.upper}")
    print(f"contour_points: {len(result.contour_points)}")
    print(f"centerline_csv: {csv_path}")
    print(f"mask: {mask_path}")
    print(f"edges: {edge_path}")
    print(f"overlay: {overlay_path}")
    print(f"debug_grid: {grid_path}")
    print("centerline_points_xy:")
    for index, (x, y) in enumerate(result.som_points):
        print(f"  {index}: ({x:.2f}, {y:.2f})")

    if args.show:
        cv2.imshow("centerline", make_debug_grid(result))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
