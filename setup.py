from setuptools import setup

setup(
      name='imbalanced_metrics',
      version='0.0.3',
      description='Perfromance metrics for imbalanced classification and imbalanced regression tasks',
      long_description = open('README.md').read(),
      long_description_content_type = "text/markdown",
      url='https://github.com/paobranco/ImbalanceMetrics',
      author='Sadid Rafsun Tulon, Jean-Gabriel Gaudreault, Paula Branco',
      author_email = 'stulo080@uottawa.ca, j.gaudreault@uottawa.ca, pbranco@uottawa.ca',
      classifiers = [
        'Programming Language :: Python :: 3',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)'
      ],
      keywords = [
        'imbalanced learning',
        'classification',
        'regression',
        'metrics'
      ],
      packages=['imbalanced_metrics'],
      
      install_requires=[
          'scikit-learn','numpy','smogn', 'pandas', 'numpy'
      ],
      include_package_data = True,
      zip_safe=False
      )