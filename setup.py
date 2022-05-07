from setuptools import find_packages, setup

REQUIREMENTS_TXT = 'requirements.txt'

with open(REQUIREMENTS_TXT, 'r', encoding='utf-8') as f:
    required = f.read().splitlines()

setup(
    name='ML_in_prod_HW1',
    packages=find_packages(),
    version="0.1.0",
    description='Homeworks for the course "ML in production" from VK Education (Technopark)',
    author='Borisov Ivan',
    license='VK education (Technopark)',
    install_requires=required,
)