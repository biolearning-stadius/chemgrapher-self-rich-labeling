import setuptools

setuptools.setup(
    name="ChemGrapher",
    version="0.1.0",
    description="ChemGrapher package",
    long_description="Optical Graph Recognition of Chemical Compounds by Deep Learning",
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "pandas", "sklearn", "requests", "bs4", "scikit-image", "opencv-python"],
    )
