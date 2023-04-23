from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT='-e .'

def get_requirements(file_path:str) -> List[str]:
    """
    This function should return a list of requirements
    """
    requirements =[]
    with open(file=file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements =[i.replace("\n","") for i in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


# setup Metadata

setup(
name='Ml_Journ',
version='0.0.1',
author='Hansel',
author_email='hanseldias919@gmail.com',
packages=find_packages(),
install_requires=get_requirements("requirements.txt")
)
