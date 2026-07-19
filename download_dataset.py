from __future__ import annotations

from pathlib import Path

OFFICIAL_PAGE = "https://datasets.simula.no/kvasir-seg/"


def main() -> None:
    target = Path("data/Kvasir-SEG")
    print("Kvasir-SEG must be obtained from its official research dataset page:")
    print(OFFICIAL_PAGE)
    print("\nAfter downloading and extracting, arrange it as:")
    print(target / "images")
    print(target / "masks")
    print("The dataset is not redistributed by this repository.")


if __name__ == "__main__":
    main()
