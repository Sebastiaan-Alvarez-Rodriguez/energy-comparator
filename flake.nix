{
  description = "energy-comparator";
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils, ... }: flake-utils.lib.eachDefaultSystem (system:
    let
      pkgs = import nixpkgs {
        inherit system;
      };

      buildPythonPackages = pkgs.python314Packages;

      # Our build
      energy-comparator = buildPythonPackages.buildPythonApplication {
        pname = "energy-comparator";
        version = "0.0.1";

        meta = {
          homepage = "https://github.com/Sebastiaan-Alvarez-Rodriguez/energy-comparator";
          description = "Compares energy and gas contracts to find optimal contracts.";
        };
        src = ./.;

        propagatedBuildInputs = [ buildPythonPackages.numpy buildPythonPackages.pandas buildPythonPackages.scipy ];

        # By default tests are executed, but we don't want to.
        dontUseSetuptoolsCheck = true;
      };
    in rec {
      apps.default = flake-utils.lib.mkApp {
        drv = packages.default;
      };
      packages.default = energy-comparator;
      devShells.default = pkgs.mkShell rec {
        packages = [ buildPythonPackages.numpy buildPythonPackages.pandas buildPythonPackages.scipy ];
      };
    }
  );
}

