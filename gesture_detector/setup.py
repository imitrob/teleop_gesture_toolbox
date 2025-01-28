from setuptools import setup
from glob import glob
import os 

package_name = 'gesture_detector'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    data_files=[
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        (os.path.join('share', package_name, 'saved_models'), glob('saved_models/*')),
    ],
    zip_safe=True,
    maintainer='Petr Vanc',
    maintainer_email='petr.vanc@cvut.cz',
    description='',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'static_detector = gesture_detector.gesture_classification.main_sample_thread:run_static',
            'dynamic_detector = gesture_detector.gesture_classification.main_sample_thread:run_dynamic',
            'custom_detector = gesture_detector.gesture_classification.main_sample_thread:run_from_rosparam',
            'leap = gesture_detector.hand_processing.leap:ros_run',
            'realsense = gesture_detector.hand_processing.realsense:main',
            'gesture_detect = gesture_detector.gesture_detect:main',
            'hand_marker_pub = gesture_detector.live_display.hand_marker_pub:main',
            'leap_backend = gesture_detector.hand_processing.leap_backend:main'
        ],
    },
)




