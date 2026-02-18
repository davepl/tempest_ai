import re
from typing import List, Tuple

import win32api
import win32con
import win32gui

# Tempest/MAME is 4:3; use this when scoring layouts so tiles are as usable as possible.
TARGET_CONTENT_ASPECT = 4.0 / 3.0
MAME_TITLE_RE = re.compile(r"\bmame\b", re.IGNORECASE)


def _split_count(total: int, buckets: int) -> List[int]:
    base = total // buckets
    extra = total % buckets
    return [base + 1 if i < extra else base for i in range(buckets)]


def _fit_area(width: int, height: int, aspect: float) -> float:
    if width <= 0 or height <= 0:
        return 0.0
    fit_w = min(width, int(round(height * aspect)))
    fit_h = int(round(fit_w / aspect))
    return float(max(0, fit_w) * max(0, fit_h))


def _is_window_eligible(hwnd: int) -> bool:
    if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
        return False
    # Top-level/ownerless window only.
    if win32gui.GetWindow(hwnd, win32con.GW_OWNER):
        return False
    title = (win32gui.GetWindowText(hwnd) or "").strip()
    if not title:
        return False
    return bool(MAME_TITLE_RE.search(title))


def _get_work_area() -> Tuple[int, int, int, int]:
    # Avoid placing windows under the taskbar.
    left, top, right, bottom = win32gui.SystemParametersInfo(win32con.SPI_GETWORKAREA)
    return int(left), int(top), int(right), int(bottom)


def _choose_layout(count: int, width: int, height: int) -> Tuple[int, List[int]]:
    # Choose row count that maximizes minimum 4:3 usable area per tile.
    best_rows = 1
    best_row_counts = [count]
    best_min_fit = -1.0
    best_avg_fit = -1.0

    for rows in range(1, count + 1):
        row_counts = _split_count(count, rows)

        y_edges = [round(i * height / rows) for i in range(rows + 1)]
        fits = []
        for r, cols in enumerate(row_counts):
            row_h = y_edges[r + 1] - y_edges[r]
            x_edges = [round(i * width / cols) for i in range(cols + 1)]
            for c in range(cols):
                tile_w = x_edges[c + 1] - x_edges[c]
                fits.append(_fit_area(tile_w, row_h, TARGET_CONTENT_ASPECT))

        min_fit = min(fits) if fits else 0.0
        avg_fit = (sum(fits) / len(fits)) if fits else 0.0

        if (
            min_fit > best_min_fit
            or (min_fit == best_min_fit and avg_fit > best_avg_fit)
            or (min_fit == best_min_fit and avg_fit == best_avg_fit and rows < best_rows)
        ):
            best_rows = rows
            best_row_counts = row_counts
            best_min_fit = min_fit
            best_avg_fit = avg_fit

    return best_rows, best_row_counts


def tile_windows() -> None:
    windows: List[int] = []

    def enum_window(hwnd, _):
        if _is_window_eligible(hwnd):
            windows.append(hwnd)

    win32gui.EnumWindows(enum_window, None)

    # Stable order by title then hwnd.
    windows.sort(key=lambda h: ((win32gui.GetWindowText(h) or "").lower(), h))

    count = len(windows)
    if count == 0:
        print("No eligible windows found with 'MAME' in the title.")
        return

    left, top, right, bottom = _get_work_area()
    work_w = max(1, right - left)
    work_h = max(1, bottom - top)

    rows, row_counts = _choose_layout(count, work_w, work_h)
    y_edges = [round(i * work_h / rows) for i in range(rows + 1)]

    print(
        f"Work area {work_w}x{work_h}, windows={count}, rows={rows}, "
        f"distribution={row_counts}"
    )

    flags = win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE
    idx = 0
    for r, cols in enumerate(row_counts):
        row_y = top + y_edges[r]
        row_h = y_edges[r + 1] - y_edges[r]
        x_edges = [round(i * work_w / cols) for i in range(cols + 1)]

        for c in range(cols):
            if idx >= count:
                break
            hwnd = windows[idx]
            idx += 1

            if not win32gui.IsWindow(hwnd):
                continue

            x = left + x_edges[c]
            y = row_y
            w = x_edges[c + 1] - x_edges[c]
            h = row_h

            try:
                win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, x, y, w, h, flags)
            except win32gui.error as e:
                title = win32gui.GetWindowText(hwnd)
                print(f"Failed to tile {hwnd} '{title}': {e}")


if __name__ == "__main__":
    tile_windows()
