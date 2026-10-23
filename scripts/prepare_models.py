import scipy  # noqa: F401
from paddleocr import PaddleOCR


def main() -> None:
    PaddleOCR(use_angle_cls=True, lang="en")


if __name__ == "__main__":
    main()
