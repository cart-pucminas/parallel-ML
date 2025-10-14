{
  pkgs ? import <nixpkgs> { },
}:
pkgs.mkShell {
  buildInputs = with pkgs; [
    gcc
    gdb
    bear
    pkg-config
    valgrind
    linuxKernel.packages.linux_zen.perf
  ];

  shellHook = ''
    fish
  '';
}
