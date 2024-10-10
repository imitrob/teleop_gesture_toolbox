from setuptools import setup
from glob import glob
import os 

package_name = 'scene_getter'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    data_files=[
    ],
    zip_safe=True,
    maintainer='Petr Vanc',
    maintainer_email='petr.vanc@cvut.cz',
    description='',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mocked_scene = scene_getter.scene_makers.mocked_scene_maker:main',
        ],
    },
)




