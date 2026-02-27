from .gzip import compress as gzip_compress, decompress as gzip_decompress
from .zip import ZipReader, ZipWriter

__all__ = ["gzip_compress", "gzip_decompress", "ZipReader", "ZipWriter"]
