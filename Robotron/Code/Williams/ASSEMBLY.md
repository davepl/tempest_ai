# Williams 6809 Assembly Notes

The source in this folder uses legacy Williams assembler syntax (`LIB`, `IFC/IFNC`, `OPT`, macro params like `&1`) that modern assemblers do not accept directly.

## Recommended toolchain

- Install with Homebrew:
  - `brew install lwtools asm6809`
- The working assembler path here is `lwasm` (from `lwtools`) with a compatibility preprocess pass.

## Build script

Use:

```bash
cd Robotron/Code/Williams
./assemble_williams.sh RRB10.ASM
```

Optional args:

```bash
./assemble_williams.sh TARGET.ASM OUTPUT.bin A_VALUE
```

- `A_VALUE` feeds legacy `&A` conditionals (`IFC/IFNC`).
- Default output is under `.build/lwasm/`.

## Outputs

For target `RRB10.ASM`, script emits:

- `.build/lwasm/RRB10.bin`
- `.build/lwasm/RRB10.lst`
- `.build/lwasm/RRB10.map`

## Notes

- Some files are include-only support modules and are not expected to assemble standalone (for example `RRTEST2.ASM`, `RRHX4.ASM`, `RRLOGD.ASM`, `RRSCRIPT.ASM`, `RRET.ASM`).
- Assemble top-level modules that include them (`RRTEST1.ASM`, `RRLOG.ASM`, `RRTEXT.ASM`, etc.).

## ROM link script

Use `link_williams_rom.sh` to build and link a full 64K ROM image from top-level modules.

Examples:

```bash
cd Robotron/Code/Williams
./link_williams_rom.sh
./link_williams_rom.sh --variant release6
./link_williams_rom.sh --variant jap --out .build/lwasm/robotron_jap_64k.bin
./link_williams_rom.sh --variant both --split 0x1000
./link_williams_rom.sh --mame-robotron
./link_williams_rom.sh --mame-robotron --mame-zip ../../roms/robotron_williams.zip
./link_williams_rom.sh --variant release6 --mame-robotron
```

Behavior:

- Rebuilds top-level modules to S-record format (address-preserving)
- Merges them into a single 64K binary image (default fill `0xFF`)
- Optionally applies patch set modules:
  - `release6` -> `RRELESE6.ASM`
  - `jap` -> `JAPDATA.ASM`
  - `both` -> both patch sets in that order
- Optional `--split` writes chunked files for burner/MAME workflows
- `--mame-robotron` emits the exact MAME filenames into `Robotron/roms/robotron/`:
  - `2084_rom_1b_3005-13.e4` ... `2084_rom_12b_3005-24.e7`
  - and tries to source:
    - `video_sound_rom_3_std_767.ic12`
    - `decoder_rom_4.3g`
    - `decoder_rom_6.3c`
    from existing `Robotron/roms/robotron.zip` if present.
- `--mame-zip` optionally packages the staged MAME files into a zip.
- `--mame-robotron` also deploys the staged set to `~/mame/roms/robotron2/`.
- In `--mame-robotron` mode, the CPU split map matches the real board layout:
  - `0x0000..0x8FFF` -> ROMs 1..9
  - `0xD000..0xFFFF` -> ROMs 10..12
- In `--mame-robotron` mode with `--variant none`, the script applies canonical parent-set fixups:
  - release bytes at `0x26B1..0x26B8` and `0xDF34..0xDF3A`
  - IRQ vector mirror at `0xFFF8..0xFFF9`
  - ROMTAB checksum/fudger bytes around `0xFFB5..0xFFD6`

Output defaults:

- Full image: `.build/lwasm/robotron_williams_64k.bin`
- Optional split dir: `.build/lwasm/<stem>_split/`

Important:

- `--mame-robotron --variant none` targets the MAME parent `robotron` checksums (`mame -verifyroms robotron` should pass).
- Non-`none` variants (`release6`, `jap`, `both`) intentionally produce different bytes and will not match parent-set checksums.
