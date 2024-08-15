from setuptools import setup

package_name = 'pointing_object_selection'

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
            'selector_node = pointing_object_selection.deictic_lib:run_node_default',
        ],
    },
)




