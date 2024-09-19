from setuptools import find_packages,setup
from typing import List


def get_requirements(file_path)->List[str]:
    requires = []
    hyphen_e_dot = '-e .'
    with open(file_path) as req_file:
        requires = req_file.readlines()
        requires = [i.replace('\n','') for i in requires]
        requires = [i for i in requires if not i.startswith('#')]
    if hyphen_e_dot in requires:
        requires.remove(hyphen_e_dot)
    return requires


        
setup( 
    name='DigitClassifier',
    version= 1.0,
    description='Classify Digit by Write it!',
    author='Kenil KB',
    author_email='bhikadiyakenil2611@gmail.com',
    packages = find_packages(),
    requires=get_requirements('./requirements.txt')
)

