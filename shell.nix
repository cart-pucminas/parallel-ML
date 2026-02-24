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
    (python3.withPackages (ps: [
      ps.numpy
      ps.matplotlib
    ]))
  ];
}
