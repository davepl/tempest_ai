#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/.build/lwasm"
PRE_DIR="$BUILD_DIR/src"

usage() {
    cat <<'EOF'
Usage: assemble_williams.sh [TARGET.ASM] [OUTPUT.bin] [A_VALUE]

Builds Williams 6809 source with lwasm after a compatibility preprocess pass.

Arguments:
  TARGET.ASM   Source file inside this directory (default: RRB10.ASM)
  OUTPUT.bin   Output binary path (default: .build/lwasm/<TARGET>.bin)
  A_VALUE      Value for legacy &A conditionals (default: empty)

Examples:
  ./assemble_williams.sh RRB10.ASM
  ./assemble_williams.sh RRS22.ASM /tmp/ros.bin NOL
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

TARGET="${1:-RRB10.ASM}"
A_VALUE="${3:-}"

if ! command -v lwasm >/dev/null 2>&1; then
    echo "error: lwasm not found. Install with: brew install lwtools" >&2
    exit 1
fi

if [[ ! -f "$SCRIPT_DIR/$TARGET" ]]; then
    echo "error: target not found: $SCRIPT_DIR/$TARGET" >&2
    exit 1
fi

mkdir -p "$PRE_DIR"

for src in "$SCRIPT_DIR"/*.ASM; do
    base="$(basename "$src")"
    dst="$PRE_DIR/$base"
    perl - "$src" "$dst" "$A_VALUE" <<'PERL'
use strict;
use warnings;

my ($in, $out, $a_value) = @ARGV;
$a_value = '' if !defined $a_value;
my $a_upper = uc($a_value);

open(my $fh, '<', $in) or die "open $in: $!";
open(my $of, '>', $out) or die "open $out: $!";

my @active = (1);
my @cond_true;

while (my $line = <$fh>) {
    my $trim = $line;
    $trim =~ s/^\s+//;
    my $opcol = ($line =~ /^\s+/) ? 1 : 0;

    if ($opcol && $trim =~ /^IFNC\s+&A\s*,\s*([A-Za-z0-9_.$?]+)/i) {
        my $tok = uc($1);
        my $parent = $active[-1];
        my $cond = ($a_upper ne $tok) ? 1 : 0;
        push @cond_true, $cond;
        push @active, ($parent && $cond) ? 1 : 0;
        next;
    }

    if ($opcol && $trim =~ /^IFC\s+&A\s*,\s*([A-Za-z0-9_.$?]+)/i) {
        my $tok = uc($1);
        my $parent = $active[-1];
        my $cond = ($a_upper eq $tok) ? 1 : 0;
        push @cond_true, $cond;
        push @active, ($parent && $cond) ? 1 : 0;
        next;
    }

    if ($opcol && $trim =~ /^ELSE\b/i) {
        die "ELSE without IF in $in\n" if !@cond_true;
        my $parent = $active[-2];
        my $cond = $cond_true[-1] ? 0 : 1;
        $active[-1] = ($parent && $cond) ? 1 : 0;
        next;
    }

    if ($opcol && $trim =~ /^ENDIF\b/i) {
        die "ENDIF without IF in $in\n" if !@cond_true;
        pop @cond_true;
        pop @active;
        next;
    }

    next if !$active[-1];

    if ($opcol && $trim =~ /^LIB\s+([A-Za-z0-9_.$?]+)/i) {
        my $lib = $1;
        print {$of} qq{ INCLUDE "$lib.ASM"\n};
        next;
    }

    if ($opcol && $trim =~ /^(OPT|STTL|TTL)\b/i) {
        print {$of} '* ' . $line;
        next;
    }

    if ($opcol && $trim =~ /^SETDP\s+RAM>>8\b/i) {
        $line =~ s/^(\s*)SETDP\s+RAM>>8\b.*$/${1}SETDP \$98\n/i;
        print {$of} $line;
        next;
    }

    $line =~ s/&([0-9])/\\$1/g;
    $line =~ s/\bLDAA\b/LDA/g;
    $line =~ s/\bLDAB\b/LDB/g;
    $line =~ s/\bORAA\b/ORA/g;
    $line =~ s/\bORAB\b/ORB/g;
    $line =~ s/\bSTAA\b/STA/g;
    $line =~ s/\bSTAB\b/STB/g;
    $line =~ s/>>8/\/256/g;
    $line =~ s/>>2/\/4/g;
    $line =~ s/>>1/\/2/g;
    $line =~ s/\[0,\s*([XYSU](?:\+\+|--|\+|-)?)\]/[,${1}]/g;
    $line =~ s/\b0,\s*([XYSU])\b/,${1}/g;
    $line =~ s/\b0,([XYSU])\+/,${1}+/g;
    print {$of} $line;
}

die "unclosed IF/ENDIF in $in\n" if @cond_true;
PERL
done

if [[ -n "${2:-}" ]]; then
    OUTPUT="$2"
else
    stem="${TARGET%.ASM}"
    OUTPUT="$BUILD_DIR/${stem}.bin"
fi

mkdir -p "$(dirname "$OUTPUT")"

LISTING="$BUILD_DIR/${TARGET%.ASM}.lst"
MAPFILE="$BUILD_DIR/${TARGET%.ASM}.map"

lwasm --6809 --6800compat -f raw -I "$PRE_DIR" \
    --list="$LISTING" --map="$MAPFILE" \
    -o "$OUTPUT" "$PRE_DIR/$TARGET"

echo "Assembled: $TARGET"
echo "Output:    $OUTPUT"
echo "Listing:   $LISTING"
echo "Map:       $MAPFILE"
