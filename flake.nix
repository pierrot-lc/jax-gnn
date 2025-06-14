{
  description = "Jax devshell";

  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
      };
    };
    cudaPackages = pkgs.cudaPackages_12_8;
    pythonPackages = pkgs.python313Packages;

    packages = [
      pkgs.uv
      pythonPackages.venvShellHook
    ];

    libs = [
      cudaPackages.cudatoolkit
      cudaPackages.cudnn
      pkgs.stdenv.cc.cc.lib
      pkgs.zlib

      # Where your local "lib/libcuda.so" lives. If you're not on NixOS,
      # you should provide the right path (likely another one).
      "/run/opengl-driver"
    ];

    shell = pkgs.mkShell {
      name = "jax-gnn";
      inherit packages;

      env = {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
        XLA_FLAGS = "--xla_gpu_cuda_data_dir=${cudaPackages.cudatoolkit}";
      };

      venvDir = "./.venv";
      postShellHook = ''
        export PATH="$PATH:${cudaPackages.cudatoolkit}/bin"  # Add ptxas to PATH.

        uv sync
        TF_CPP_MIN_LOG_LEVEL=0 python -c "from jax.extend.backend import get_backend; print('Backend:', get_backend().platform)"
      '';
    };
  in {
    devShells.${system}.default = shell;
  };
}
