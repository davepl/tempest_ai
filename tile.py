import win32gui
import win32con
import win32api
import math

def is_window_eligible(hwnd):
    # Check if visible, not minimized, has a title, and title contains "mame" (case-insensitive)
    if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
        return False
    title = win32gui.GetWindowText(hwnd)
    if not title:
        return False
    return "mame" in title.lower()

def tile_windows():
    windows = []

    def enum_window(hwnd, _):
        if is_window_eligible(hwnd):
            windows.append(hwnd)
    win32gui.EnumWindows(enum_window, None)

    count = len(windows)
    if count == 0:
        print("No eligible 'Mame' windows found.")
        return

    screen_width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
    screen_height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

    # Calculate a grid that's roughly square
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    # Ensure rows calculation doesn't lead to cols * (rows - 1) >= count
    # This can happen if count is slightly less than a perfect square number times cols
    while cols * (rows - 1) >= count and rows > 1:
            rows -= 1

    tile_width = screen_width // cols
    tile_height = screen_height // rows

    print(f"Screen: {screen_width}x{screen_height}. Found {count} windows. Tiling in {rows}x{cols} grid.")
    print(f"Tile size: {tile_width}x{tile_height}")

    for idx, hwnd in enumerate(windows):
        if not win32gui.IsWindow(hwnd):
            print(f"Window {hwnd} disappeared before tiling.")
            continue
        row = idx // cols
        col = idx % cols
        x = col * tile_width
        y = row * tile_height
        # Use SWP_NOZORDER to prevent windows from changing their Z-order
        # Use SWP_NOACTIVATE to prevent windows from stealing focus
        flags = win32con.SWP_NOZORDER | win32con.SWP_NOACTIVATE
        try:
            # SetWindowPos is often more reliable than MoveWindow for positioning
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOP, x, y, tile_width, tile_height, flags)
            # Optional: Bring window to top without activating
            # win32gui.BringWindowToTop(hwnd)
        except win32gui.error as e:
            print(f"Error tiling window {hwnd} ('{win32gui.GetWindowText(hwnd)}'): {e}")
            pass  # safely ignore if window disappears or other errors occur

if __name__ == "__main__":
    tile_windows()