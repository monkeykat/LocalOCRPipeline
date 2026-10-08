from docling.utils.model_downloader import download_models
from paddleocr import PaddleOCR


def main() -> None:
    download_models()
    PaddleOCR(use_angle_cls=True, lang="en")


if __name__ == "__main__":
    main()
