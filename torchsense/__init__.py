from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("torchsense")
except PackageNotFoundError:
    __version__ = "unknown version"
