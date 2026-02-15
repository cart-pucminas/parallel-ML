{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc
    gdb
    gnumake
    bear
    pkg-config
    valgrind
    perf
  ];
}
