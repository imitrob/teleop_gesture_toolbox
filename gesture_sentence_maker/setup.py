from setuptools import setup
from glob import glob

package_name = 'gesture_sentence_maker'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    zip_safe=True,
    maintainer='Petr Vanc',
    maintainer_email='petr.vanc@cvut.cz',
    description='',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sentence_maker = gesture_sentence_maker.gesture_processor:main',
            'deictic_processor = gesture_sentence_maker.gesture_deictic_processor_standalone:main'
        ],
    },
)




