from setuptools import setup

package_name = 'gesture_meaning'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    package_data={package_name: ['links.yaml', 'link_game.html']},
    install_requires=['setuptools'],
    data_files=[
    ],
    zip_safe=True,
    maintainer='Petr Vanc',
    maintainer_email='petr.vanc@cvut.cz',
    description='Gesture name to action name mapping, from the user links.',
    license='Apache 2.0',
    tests_require=['pytest'],
)




