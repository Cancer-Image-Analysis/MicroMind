from setuptools import setup


setup(name='cania-utils',
      version='0.2.2',
      description='All utils for Cancer Image Analysis python package',
      long_description='',
      classifiers=[
        'Development Status :: 1 - Planning',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Intended Audience :: Healthcare Industry',
      ],
      keywords='cancer artificial-intelligence computer-vision',
      url='https://github.com/Cancer-Image-Analysis/cania-utils',
      author='Kevin Cortacero',
      author_email='kevin.cortacero@inserm.fr',
      license='MIT',
      packages=['cania_utils'],
      install_requires=[
          'numpy',
          'tifffile',
          'pandas',
          'opencv-python',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
