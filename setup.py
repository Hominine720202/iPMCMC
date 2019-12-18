# To make majelan_tools as a real python module that can be installed via pip

from setuptools import setup, find_packages
from ipmcmc import __version__


setup(name='ipmcmc',
      version=__version__,
      description='Python implementation of the ipmcmc algorithm',
      authors=['Corentin Ambroise', 'Luis Montero'],
      authors_email=['corentin.ambroise@u-psud.fr', 'luis.montero@u-psud.fr'],
      url="https://github.com/fd0r/iPMCMC/",  # TODO: Add link to repository
      packages=find_packages(),
      install_requires=['numpy', 'scipy', 'tqdm', 'filterpy'],
      extra_requires=dict()
      )
