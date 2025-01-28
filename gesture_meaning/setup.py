from setuptools import setup

package_name = 'gesture_meaning'

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
            'gesture_meaning_service = gesture_meaning.gesture_meaning_service:main',
            'compound_gesture_meaning = gesture_meaning.gesture_meaning_service:compound_gesture_meaning',
            'compound_gesture_user_meaning = gesture_meaning.gesture_meaning_service:compound_gesture_user_meaning',
        ],
    },
)




