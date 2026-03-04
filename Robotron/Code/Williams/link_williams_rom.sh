#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/.build/lwasm"
PRE_DIR="$BUILD_DIR/src"

usage() {
    cat <<'USAGE'
Usage: link_williams_rom.sh [--out FILE] [--variant none|release6|jap|both] [--split SIZE]
                            [--mame-robotron] [--mame-zip FILE] [--no-mame-parent-fixups]

Builds the Williams Robotron source into a 64K ROM image by:
1) Assembling top-level modules to S-record (address-aware)
2) Merging records into a single 0x0000-0xFFFF image (default fill=0xFF)
3) Applying optional patch sets (RRELESE6/JAPDATA)
4) Optionally splitting into fixed-size chunks

Options:
  --out FILE                Output ROM image path
                            default: .build/lwasm/robotron_williams_64k.bin
  --variant MODE            Patch set to apply: none, release6, jap, both
                            default: none
  --split SIZE              Split final ROM into chunk SIZE bytes (hex like 0x1000 or decimal)
  --mame-robotron           Stage MAME-style Robotron ROM files into Robotron/roms/robotron
  --mame-zip FILE           Create a zip file from staged MAME files (use with --mame-robotron)
  --no-mame-parent-fixups   Disable canonical MAME parent fixups in --mame-robotron mode
  -h, --help                Show this help

Examples:
  ./link_williams_rom.sh
  ./link_williams_rom.sh --variant release6
  ./link_williams_rom.sh --variant jap --out .build/lwasm/robotron_jap.bin
  ./link_williams_rom.sh --variant both --split 0x1000
  ./link_williams_rom.sh --mame-robotron
  ./link_williams_rom.sh --mame-robotron --mame-zip ../../roms/robotron_williams.zip
USAGE
}

OUT_FILE="$BUILD_DIR/robotron_williams_64k.bin"
VARIANT="none"
SPLIT_SIZE=""
MAME_ROBOTRON=0
MAME_ZIP=""
MAME_PARENT_FIXUPS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out)
            OUT_FILE="$2"
            shift 2
            ;;
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --split)
            SPLIT_SIZE="$2"
            shift 2
            ;;
        --mame-robotron)
            MAME_ROBOTRON=1
            shift
            ;;
        --mame-zip)
            MAME_ZIP="$2"
            shift 2
            ;;
        --no-mame-parent-fixups)
            MAME_PARENT_FIXUPS=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$VARIANT" in
    none|release6|jap|both) ;;
    *)
        echo "error: invalid --variant '$VARIANT' (use none|release6|jap|both)" >&2
        exit 1
        ;;
esac

if ! command -v lwasm >/dev/null 2>&1; then
    echo "error: lwasm not found. Install with: brew install lwtools" >&2
    exit 1
fi

if [[ ! -x "$SCRIPT_DIR/assemble_williams.sh" ]]; then
    echo "error: missing helper script: $SCRIPT_DIR/assemble_williams.sh" >&2
    exit 1
fi

if [[ -n "$MAME_ZIP" && "$MAME_ROBOTRON" -ne 1 ]]; then
    echo "error: --mame-zip requires --mame-robotron" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"
mkdir -p "$(dirname "$OUT_FILE")"

TOP_LEVEL_MODULES=(
    RRH11
    RRC11
    RRB10
    RRG23
    RRP8
    RRT2
    RRDX2
    RRTK4
    RRM1
    RRX7
    RRTEXT
    RRTESTB
    RRLOG
    RRS22
    RRTABLE
    RRTESTC
    RRSET
    RRTEST1
)

PATCH_MODULES=()
case "$VARIANT" in
    release6)
        PATCH_MODULES=(RRELESE6)
        ;;
    jap)
        PATCH_MODULES=(JAPDATA)
        ;;
    both)
        PATCH_MODULES=(RRELESE6 JAPDATA)
        ;;
esac

assemble_to_srec() {
    local mod="$1"
    local src="$mod.ASM"
    local srec="$BUILD_DIR/$mod.s19"

    "$SCRIPT_DIR/assemble_williams.sh" "$src" >/dev/null
    lwasm --6809 --6800compat -f srec -I "$PRE_DIR" \
        -o "$srec" "$PRE_DIR/$src"
    echo "$srec"
}

echo "Assembling top-level modules..."
SREC_FILES=()
for mod in "${TOP_LEVEL_MODULES[@]}"; do
    srec_path="$(assemble_to_srec "$mod")"
    SREC_FILES+=("$srec_path")
    printf "  %-8s -> %s\n" "$mod" "$(basename "$srec_path")"
done

PATCH_SRECS=()
if [[ ${#PATCH_MODULES[@]} -gt 0 ]]; then
    echo "Assembling patch modules ($VARIANT)..."
    for mod in "${PATCH_MODULES[@]}"; do
        srec_path="$(assemble_to_srec "$mod")"
        PATCH_SRECS+=("$srec_path")
        printf "  %-8s -> %s\n" "$mod" "$(basename "$srec_path")"
    done
fi

PY_ARGS=("$OUT_FILE" "$SPLIT_SIZE" "$VARIANT" "$MAME_ROBOTRON" "$MAME_PARENT_FIXUPS")
PY_ARGS+=("${SREC_FILES[@]}")
PY_ARGS+=("--")
if [[ ${#PATCH_SRECS[@]} -gt 0 ]]; then
    PY_ARGS+=("${PATCH_SRECS[@]}")
fi

python3 - "${PY_ARGS[@]}" <<'PY'
import sys
from pathlib import Path

out_path = Path(sys.argv[1])
split_size_raw = sys.argv[2]
variant = sys.argv[3]
mame_robotron = sys.argv[4] == "1"
mame_parent_fixups = sys.argv[5] == "1"
args = sys.argv[6:]

sep = args.index("--")
module_srecs = [Path(p) for p in args[:sep]]
patch_srecs = [Path(p) for p in args[sep + 1:]]

if split_size_raw:
    split_size = int(split_size_raw, 0)
    if split_size <= 0:
        raise SystemExit("error: --split must be > 0")
else:
    split_size = None

mem = bytearray([0xFF] * 0x10000)
owners = [None] * 0x10000


def parse_srec_line(line: str):
    line = line.strip()
    if not line.startswith("S") or len(line) < 4:
        return None
    rectype = line[1]
    if rectype not in {"1", "2", "3"}:
        return None

    count = int(line[2:4], 16)
    if rectype == "1":
        addr_len = 2
    elif rectype == "2":
        addr_len = 3
    else:
        addr_len = 4

    addr_hex_len = addr_len * 2
    addr_start = 4
    addr_end = addr_start + addr_hex_len
    addr = int(line[addr_start:addr_end], 16)

    data_len = count - addr_len - 1
    data_start = addr_end
    data_end = data_start + data_len * 2
    data = bytes.fromhex(line[data_start:data_end]) if data_len > 0 else b""
    return addr, data


def merge_srec(path: Path, allow_override: bool):
    with path.open("r", encoding="ascii", errors="ignore") as fh:
        for line in fh:
            parsed = parse_srec_line(line)
            if parsed is None:
                continue
            addr, data = parsed
            for i, b in enumerate(data):
                a = addr + i
                if not (0 <= a <= 0xFFFF):
                    raise SystemExit(f"error: address out of range in {path}: 0x{a:05X}")
                cur = mem[a]
                if allow_override:
                    mem[a] = b
                    owners[a] = path.name
                    continue
                if cur != 0xFF and cur != b:
                    prev = owners[a] if owners[a] else "unknown"
                    raise SystemExit(
                        f"error: overlap conflict at 0x{a:04X}: {prev}=0x{cur:02X}, {path.name}=0x{b:02X}"
                    )
                mem[a] = b
                owners[a] = path.name


def page_sum(page: int) -> int:
    start = page << 12
    return sum(mem[start : start + 0x1000]) & 0xFF


def apply_mame_parent_fixups():
    mem[0x26B1:0x26B9] = b"\x55" * 8
    mem[0xDF34:0xDF3B] = b"\x0A" * 7

    # Canonical parent IRQ vector mirrors EFF8/EFF9 into FFF8/FFF9.
    mem[0xFFF8] = mem[0xEFF8]
    mem[0xFFF9] = mem[0xEFF9]

    # Populate ROMTAB sums for installed pages in the MAME parent set.
    stuffed_pages = {0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x8, 0xD, 0xE, 0xF}
    table_base = 0xFFB5
    for page in range(0x10):
        sum_addr = table_base + page * 2 + 1
        mem[sum_addr] = page_sum(page) if page in stuffed_pages else 0x00

    # Match canonical parent ROMTAB + fudger behavior for F000 page.
    mem[0xFFD4] = 0x01
    current = page_sum(0xF)
    delta = (mem[0xFFD4] - current) & 0xFF
    mem[0xFFD6] = (mem[0xFFD6] + delta) & 0xFF

    final_sum = page_sum(0xF)
    if final_sum != mem[0xFFD4]:
        raise SystemExit(
            f"error: failed to set F000 checksum (have 0x{final_sum:02X}, want 0x{mem[0xFFD4]:02X})"
        )


for srec in module_srecs:
    merge_srec(srec, allow_override=False)

for srec in patch_srecs:
    merge_srec(srec, allow_override=True)

applied_mame_parent_fixups = False
if mame_robotron and mame_parent_fixups and variant == "none":
    apply_mame_parent_fixups()
    applied_mame_parent_fixups = True

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_bytes(mem)

used = sum(1 for x in owners if x is not None)
print(f"Linked image: {out_path}")
print(f"Bytes populated: {used} / 65536 ({used/65536:.1%})")
if patch_srecs:
    print(f"Applied patches ({variant}): {', '.join(p.name for p in patch_srecs)}")
else:
    print("Applied patches: none")

if mame_robotron and mame_parent_fixups and variant != "none":
    print(f"Canonical MAME parent fixups skipped for variant '{variant}'")
elif applied_mame_parent_fixups:
    print("Applied canonical MAME parent fixups: release bytes, vectors, ROMTAB checksums")

if split_size is not None:
    stem = out_path.stem
    split_dir = out_path.parent / f"{stem}_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, 0x10000, split_size):
        chunk = mem[i : min(i + split_size, 0x10000)]
        end = i + len(chunk) - 1
        name = f"{stem}_{i:04X}_{end:04X}.bin"
        (split_dir / name).write_bytes(chunk)
    print(f"Split chunks written to: {split_dir} (size={split_size} bytes)")
PY

if [[ "$MAME_ROBOTRON" -eq 1 ]]; then
    MAME_DIR="$(cd "$SCRIPT_DIR/../../roms" && pwd)/robotron"
    SOURCE_ZIP="$(cd "$SCRIPT_DIR/../../roms" && pwd)/robotron.zip"
    mkdir -p "$MAME_DIR"

    CPU_FILES=(
        "2084_rom_1b_3005-13.e4"
        "2084_rom_2b_3005-14.c4"
        "2084_rom_3b_3005-15.a4"
        "2084_rom_4b_3005-16.e5"
        "2084_rom_5b_3005-17.c5"
        "2084_rom_6b_3005-18.a5"
        "2084_rom_7b_3005-19.e6"
        "2084_rom_8b_3005-20.c6"
        "2084_rom_9b_3005-21.a6"
        "2084_rom_10b_3005-22.a7"
        "2084_rom_11b_3005-23.c7"
        "2084_rom_12b_3005-24.e7"
    )
    CPU_ADDRS=(
        0x0000 0x1000 0x2000 0x3000 0x4000 0x5000 0x6000 0x7000 0x8000
        0xD000 0xE000 0xF000
    )

    echo "Staging MAME CPU ROM files in: $MAME_DIR"
    for i in "${!CPU_FILES[@]}"; do
        offset=$((CPU_ADDRS[$i]))
        dd if="$OUT_FILE" of="$MAME_DIR/${CPU_FILES[$i]}" bs=1 skip="$offset" count=$((0x1000)) status=none
    done

    EXTRA_FILES=(
        "video_sound_rom_3_std_767.ic12"
        "decoder_rom_4.3g"
        "decoder_rom_6.3c"
    )

    if [[ -f "$SOURCE_ZIP" ]]; then
        for f in "${EXTRA_FILES[@]}"; do
            if ! unzip -p "$SOURCE_ZIP" "$f" > "$MAME_DIR/$f" 2>/dev/null; then
                echo "warning: could not extract $f from $SOURCE_ZIP" >&2
            fi
        done
    else
        for f in "${EXTRA_FILES[@]}"; do
            if [[ ! -f "$MAME_DIR/$f" ]]; then
                echo "warning: missing $f (no source zip found at $SOURCE_ZIP)" >&2
            fi
        done
    fi

    if [[ -n "$MAME_ZIP" ]]; then
        mkdir -p "$(dirname "$MAME_ZIP")"
        ZIP_PATH="$(cd "$(dirname "$MAME_ZIP")" && pwd)/$(basename "$MAME_ZIP")"
        (
            cd "$MAME_DIR"
            zip -q -9 "$ZIP_PATH" "${CPU_FILES[@]}" "${EXTRA_FILES[@]}" 2>/dev/null || \
            zip -q -9 "$ZIP_PATH" "${CPU_FILES[@]}"
        )
        echo "Created MAME zip: $ZIP_PATH"
    fi

fi

echo "Done."
