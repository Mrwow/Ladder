import setuptools
setuptools.setup(
    name='ladder',
    version='0.1.1',
    description='A software to label images, detect objects and deploy models recurrently',
    url='https://github.com/Mrwow/Ladder',
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: MacOS :: MacOS X"
        "Operating System :: Microsoft :: Windows :: Windows 10"
    ],
    author='Joe Tang',
    author_email='zhou.tang@wsu.edu',
    license='GPLv3',
    packages=['ladder'],
    # include_package_data=True,
    install_requires=['numpy', 'pandas','tqdm',
                      # data processing
                      'qtpy!=1.11.2', 'PyQt5!=5.15.3, !=5.15.4', 'pyside2',
                      # GUI
                      'pillow', 'opencv-python', 'matplotlib', 'imgviz>=0.11', 'seaborn',
                      # image processing
                      'torch==1.9.0', 'tensorboard', 'torchvision==0.10.0'
                      ],
    entry_points={
                 "console_scripts": ["ladder=ladder.__main__:main"],
             },
)
