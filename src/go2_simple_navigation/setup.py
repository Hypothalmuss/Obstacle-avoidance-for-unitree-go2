from setuptools import setup
import os
from glob import glob

package_name = 'go2_simple_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nadim',
    maintainer_email='mohamednadimtouil@gmail.com',
    description='Simple navigation package for Go2 robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'go2_simple_navigation_node = go2_simple_navigation.go2_simple_navigation_node:main',
            'simple_avoidance = go2_simple_navigation.simple_avoidance:main',
            'walk = go2_simple_navigation.walk:main',
            'avoid_dwb = go2_simple_navigation.avoid_dwb:main',
            'goal_avoid = go2_simple_navigation.goal_avoid:main',
  
        ],
    },
)

