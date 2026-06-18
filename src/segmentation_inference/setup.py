from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'segmentation_inference'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ronak_Viradiya',
    maintainer_email='ronak.viradiya84@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'inference_node = segmentation_inference.inference_node:main',
            'rangenetpp_inference_node = segmentation_inference.rangenetpp_inference_node:main',
        ],
    },
)
