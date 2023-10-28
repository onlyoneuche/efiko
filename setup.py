from setuptools import setup, find_packages

SETUP_INSTALL_TRIGGER = '-e .'


def get_requirements(path):
    """return a list of requirements from a file"""

    requirements = []
    with open(path) as f:
        requirements = f.readlines()
        requirements = [r.replace("/n", "") for r in requirements]

        if SETUP_INSTALL_TRIGGER in requirements:
            requirements.remove(SETUP_INSTALL_TRIGGER)
    return requirements


setup(
    name='efiko',
    version='0.0.1',
    author='Uchechukwu Emmanuel',
    author_email='onlyoneuche@gmail.com',
    description='A simple package for efiko',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
