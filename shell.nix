{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc
    gdb
    bear
    pkg-config
    valgrind
  ];
}
