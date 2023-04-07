from setuptools import setup

setup(name='imbalanced_metrics',
      version='0.1',
      description='Perfromance metrics for imbalanced classification and imbalanced regression tasks',
      url='https://github.com/paobranco/ImablanceMetrics',
      authors=['Sadid Rafsun Tulon','Jean-Gabriel Gaudreault','Paula Branco'],
      packages=['imbalanced_metrics'],
      install_requires=[
          'scikit-learn','numpy','smogn',
      ],
      zip_safe=False)