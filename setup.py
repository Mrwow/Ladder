import setuptools
setuptools.setup(
    name='ladder',
    version='0.1.1',
    description='A software to label images, detect objects and deploy models recurrently',
    url='https://github.com/Mrwow/Ladder',
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3.7.11",
        "Programming Language :: Python :: 3.8.17",
        "Programming Language :: Python :: 3.9.7",
        "Programming Language :: Python :: 3.10.12",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: MacOS :: MacOS X"
        "Operating System :: Microsoft :: Windows :: Windows 10"
    ],
    author='Joe Tang',
    author_email='zhou.tang@wsu.edu',
    license='GPLv3',
    packages=['ladder'],
    # include_package_data=True,
    install_requires=['numpy>=1.18.5', 'pandas>=1.1.4', 'scipy>=1.4.1', 'matplotlib>=3.2.2', 'seaborn>=0.11.0', 'Pillow>=7.1.2', 'opencv-python>=4.1.1', 
                      # data processing, image processing
                      'qtpy!=1.11.2', 'PyQt5!=5.15.3, !=5.15.4', 'pyside2', 'tqdm>=4.64.0', 'imgviz>=0.11', 'requests>=2.23.0','PyYAML>=5.3.1',
                      # GUI
                      'torch>=1.7.0', 'torchvision>=0.8.1', 'tensorboard>=2.4.1', "ultralytics>=8.0", "sahi>=0.11.14","albumentations>=1.3.1"
                      # deep learning
                      ],
    entry_points={
                 "console_scripts": ["ladder=ladder.__main__:main"],
             },
)
