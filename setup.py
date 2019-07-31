"""
This is the setup.py for the parameter fitting part of the pydivest model.

for developers: recommended way of installing is to run in this directory
pip install -e .
This creates a link insteaed of copying the files, so modifications in this
directory are modifications in the installed package.
"""

from setuptools import setup

setup(name="pydivest_parameters",
      version="0.0.1",
      description="fitting parameters of the pydivest model to economic data",
      url="to be added",
      author="Jakob. J. Kolb",
      author_email="kolb@pik-potsdam.de",
      license="MIT",
      packages=["pydivest_parameters"],
      include_package_data=True,
      install_requires=[
            "numpy",
            "scipy",
            "pandas",
            'sympy',
      ],
      package_data={
          "pydivest_parameters.data.Energy_Consumption": ['*.xlsx'],
          "pydivest_parameters.data.GDP": ['*.csv'],
          "pydivest_parameters.data.Labor": ['*.csv'],
          "pydivest_parameters.data.Oil_Price": ['*.xlsx'],
      },
      # see http://stackoverflow.com/questions/15869473/what-is-the-advantage-
      # of-setting-zip-safe-to-true-when-packaging-a-python-projec
      zip_safe=False)
