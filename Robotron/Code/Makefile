SHELL := /bin/bash

CODE_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
ROBOTRON_DIR := $(abspath $(CODE_DIR)/..)
WILLIAMS_DIR := $(CODE_DIR)/Williams
LINK_SCRIPT := $(WILLIAMS_DIR)/link_williams_rom.sh

VARIANT ?= none
MAME_ZIP ?= $(ROBOTRON_DIR)/roms/robotron_williams.zip

.PHONY: all help tools build roms verify clean

all: roms

help:
	@echo "Robotron build targets"
	@echo "  make tools                  Check required tools"
	@echo "  make build [VARIANT=none]   Build 64K linked image only"
	@echo "  make roms [VARIANT=none]    Build and stage MAME ROM files into Robotron/roms/robotron"
	@echo "  make verify                 Run MAME verification for robotron set"
	@echo "  make clean                  Remove generated build and ROM output"

tools:
	@command -v lwasm >/dev/null || { echo "Missing tool: lwasm (install with: brew install lwtools)"; exit 1; }
	@command -v python3 >/dev/null || { echo "Missing tool: python3"; exit 1; }
	@echo "Tool check passed."

build: tools
	@cd "$(WILLIAMS_DIR)" && "$(LINK_SCRIPT)" --variant "$(VARIANT)"

roms: tools
	@cd "$(WILLIAMS_DIR)" && "$(LINK_SCRIPT)" --variant "$(VARIANT)" --mame-robotron --mame-zip "$(MAME_ZIP)"

verify:
	@command -v mame >/dev/null || { echo "Missing tool: mame"; exit 1; }
	@mame -rompath "$(ROBOTRON_DIR)/roms" -verifyroms robotron

clean:
	rm -rf "$(WILLIAMS_DIR)/.build"
	rm -rf "$(ROBOTRON_DIR)/roms/robotron"
	rm -f "$(ROBOTRON_DIR)/roms/robotron_williams.zip"
	@echo "Cleaned generated Robotron build outputs."
