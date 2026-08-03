from setuptools import find_packages, setup

package_name = 'px4_camera_tracking'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='xueyang',
    maintainer_email='xueyang@example.com',
    description='PX4 camera tracking test package',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'camera_test_node = px4_camera_tracking.camera_test_node:main',
        'color_tracking_node = px4_camera_tracking.color_tracking_node:main',
        'image_snapshot_node = px4_camera_tracking.image_snapshot_node:main',
        'visual_orbit_node = px4_camera_tracking.visual_orbit_node:main',
    ],
  },
)
