from setuptools import setup, find_packages
import versioneer

setup_kwargs = {"include_package_data": True, "package_data": {"": ["MANIFEST.in"]}}

setup(
    name="MEGA",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Cankun Wang",
    author_email="cankun.wang@osumc.edu",
    license="MIT",
    description="MEGA is a deep learning package for identifying cancer-associated tissue-resident microbes",
    long_description="MEGA is a deep learning package for identifying cancer-associated tissue-resident microbes",
    long_description_content_type="text/markdown",
    url="https://github.com/OSU-BMBL/MEGA",
    packages=find_packages(),
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points="""
        [console_scripts]
        MEGA=MEGA.cli:main
    """,
    **setup_kwargs
)
