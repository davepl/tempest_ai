import win32gui
import win32con
import win32api
import math

def is_window_eligible(hwnd):
    return win32gui.IsWindowVisible(hwnd) and not win32gui.IsIconic(hwnd) and win32gui.GetWindowText(hwnd)

def tile_windows(aspect_ratio=2.0):
    windows = []

    def enum_window(hwnd, _):
        if is_window_eligible(hwnd):
            windows.append(hwnd)
    win32gui.EnumWindows(enum_window, None)

    count = len(windows)
    if count == 0:
        print("No eligible windows found.")
        return

    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

    # Find optimal number of columns and rows to maximize window size
    best_cols, best_rows = 1, count
    max_area = 0

    for cols in range(1, count + 1):
        rows = math.ceil(count / cols)
        tile_width = screen_width / cols
        tile_height = tile_width * aspect_ratio

        if tile_height * rows > screen_height:
            tile_height = screen_height / rows
            tile_width = tile_height / aspect_ratio

        area = tile_width * tile_height
        if area > max_area:
            max_area = area
            best_cols, best_rows = cols, rows

    tile_width = int(screen_width / best_cols)
    tile_height = int(tile_width * aspect_ratio)

    # Recheck to ensure height fits within the screen
    if tile_height * best_rows > screen_height:
        tile_height = int(screen_height / best_rows)
        tile_width = int(tile_height / aspect_ratio)

    for idx, hwnd in enumerate(windows):
        if not win32gui.IsWindow(hwnd):
            continue
        row = idx // best_cols
        col = idx % best_cols
        x = col * tile_width
        y = row * tile_height
        try:
            win32gui.MoveWindow(hwnd, x, y, tile_width, tile_height, True)
        except win32gui.error:
            pass  # safely ignore if window disappears

if __name__ == "__main__":
    tile_windows(aspect_ratio=2.0)